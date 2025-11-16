// Function: sub_D6B260
// Address: 0xd6b260
//
__int64 __fastcall sub_D6B260(__int64 a1, char *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rdx
  unsigned __int64 v9; // rcx
  unsigned __int64 v10; // rsi
  int v11; // eax
  unsigned __int64 *v12; // rdi
  unsigned __int64 v13; // rax
  __int64 result; // rax
  char *v15; // r12

  v7 = *(unsigned int *)(a1 + 8);
  v9 = *(_QWORD *)a1;
  v10 = v7 + 1;
  v11 = *(_DWORD *)(a1 + 8);
  if ( v7 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
  {
    if ( v9 > (unsigned __int64)a2 || (unsigned __int64)a2 >= v9 + 24 * v7 )
    {
      sub_D6B130(a1, v10, v7, v9, a5, a6);
      v7 = *(unsigned int *)(a1 + 8);
      v9 = *(_QWORD *)a1;
      v11 = *(_DWORD *)(a1 + 8);
    }
    else
    {
      v15 = &a2[-v9];
      sub_D6B130(a1, v10, v7, v9, a5, a6);
      v9 = *(_QWORD *)a1;
      v7 = *(unsigned int *)(a1 + 8);
      a2 = &v15[*(_QWORD *)a1];
      v11 = *(_DWORD *)(a1 + 8);
    }
  }
  v12 = (unsigned __int64 *)(v9 + 24 * v7);
  if ( v12 )
  {
    *v12 = 4;
    v13 = *((_QWORD *)a2 + 2);
    v12[1] = 0;
    v12[2] = v13;
    if ( v13 != 0 && v13 != -4096 && v13 != -8192 )
      sub_BD6050(v12, *(_QWORD *)a2 & 0xFFFFFFFFFFFFFFF8LL);
    v11 = *(_DWORD *)(a1 + 8);
  }
  result = (unsigned int)(v11 + 1);
  *(_DWORD *)(a1 + 8) = result;
  return result;
}
