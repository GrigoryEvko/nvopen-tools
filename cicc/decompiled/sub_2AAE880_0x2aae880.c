// Function: sub_2AAE880
// Address: 0x2aae880
//
__int64 __fastcall sub_2AAE880(_QWORD **a1, int *a2, __int64 a3, __int64 a4)
{
  char v4; // r9
  int v5; // r8d
  __int64 v6; // rax
  __int64 v7; // r10
  __int64 v8; // rdx
  __int64 v9; // rdi
  int v10; // eax
  unsigned int v11; // eax
  int i; // ebx
  unsigned int *v13; // rsi
  unsigned int v14; // eax

  v4 = *((_BYTE *)a2 + 4);
  v5 = *a2;
  v6 = (*a1)[5];
  v7 = *a1[1];
  if ( v4 )
  {
    v8 = *(unsigned int *)(v6 + 216);
    v9 = *(_QWORD *)(v6 + 200);
    if ( (_DWORD)v8 )
    {
      v10 = 37 * v5 - 1;
LABEL_5:
      v11 = (v8 - 1) & v10;
      for ( i = 1; ; ++i )
      {
        v13 = (unsigned int *)(v9 + 72LL * v11);
        a4 = *v13;
        if ( (_DWORD)a4 == v5 && v4 == *((_BYTE *)v13 + 4) )
        {
          v9 += 72LL * v11;
          return sub_B19060(v9 + 8, v7, v8, a4);
        }
        if ( (_DWORD)a4 == -1 && *((_BYTE *)v13 + 4) )
          break;
        v14 = i + v11;
        v11 = (v8 - 1) & v14;
      }
      v9 += 72 * v8;
    }
    return sub_B19060(v9 + 8, v7, v8, a4);
  }
  if ( v5 != 1 )
  {
    v8 = *(unsigned int *)(v6 + 216);
    v9 = *(_QWORD *)(v6 + 200);
    if ( (_DWORD)v8 )
    {
      v10 = 37 * v5;
      goto LABEL_5;
    }
    return sub_B19060(v9 + 8, v7, v8, a4);
  }
  return 1;
}
