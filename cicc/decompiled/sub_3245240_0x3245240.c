// Function: sub_3245240
// Address: 0x3245240
//
__int64 __fastcall sub_3245240(__int64 a1, char *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rdx
  unsigned __int64 v9; // rcx
  unsigned __int64 v10; // rsi
  int v11; // eax
  _QWORD *v12; // rdx
  __int64 result; // rax
  __int64 v14; // rdi
  char *v15; // r12

  v7 = *(unsigned int *)(a1 + 160);
  v9 = *(_QWORD *)(a1 + 152);
  v10 = v7 + 1;
  v11 = *(_DWORD *)(a1 + 160);
  if ( v7 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 164) )
  {
    v14 = a1 + 152;
    if ( v9 > (unsigned __int64)a2 || (unsigned __int64)a2 >= v9 + 8 * v7 )
    {
      sub_3245160(v14, v10, v7, v9, a5, a6);
      v7 = *(unsigned int *)(a1 + 160);
      v9 = *(_QWORD *)(a1 + 152);
      v11 = *(_DWORD *)(a1 + 160);
    }
    else
    {
      v15 = &a2[-v9];
      sub_3245160(v14, v10, v7, v9, a5, a6);
      v9 = *(_QWORD *)(a1 + 152);
      v7 = *(unsigned int *)(a1 + 160);
      a2 = &v15[v9];
      v11 = *(_DWORD *)(a1 + 160);
    }
  }
  v12 = (_QWORD *)(v9 + 8 * v7);
  if ( v12 )
  {
    *v12 = *(_QWORD *)a2;
    *(_QWORD *)a2 = 0;
    v11 = *(_DWORD *)(a1 + 160);
  }
  result = (unsigned int)(v11 + 1);
  *(_DWORD *)(a1 + 160) = result;
  return result;
}
