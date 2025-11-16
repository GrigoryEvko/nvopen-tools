// Function: sub_2670750
// Address: 0x2670750
//
_DWORD *__fastcall sub_2670750(_DWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v7; // eax
  __int64 v8; // rdi
  __int64 v9; // rcx
  unsigned int v10; // eax
  __int64 *v11; // rbx
  __int64 v12; // rsi
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9

  v7 = *(_DWORD *)(a2 + 248);
  v8 = *(_QWORD *)(a2 + 232);
  if ( v7 )
  {
    v9 = (unsigned int)(v7 - 1);
    v10 = v9 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
    v11 = (__int64 *)(v8 + ((unsigned __int64)v10 << 7));
    v12 = *v11;
    if ( a3 == *v11 )
    {
LABEL_3:
      *a1 = *((_DWORD *)v11 + 2);
      sub_C8CD80((__int64)(a1 + 2), (__int64)(a1 + 10), (__int64)(v11 + 2), v9, a5, a6);
      sub_C8CD80((__int64)(a1 + 14), (__int64)(a1 + 22), (__int64)(v11 + 8), v13, v14, v15);
      return a1;
    }
    a5 = 1;
    while ( v12 != -4096 )
    {
      a6 = (unsigned int)(a5 + 1);
      v10 = v9 & (a5 + v10);
      v11 = (__int64 *)(v8 + ((unsigned __int64)v10 << 7));
      v12 = *v11;
      if ( a3 == *v11 )
        goto LABEL_3;
      a5 = (unsigned int)a6;
    }
  }
  memset(a1, 0, 0x78u);
  *((_BYTE *)a1 + 36) = 1;
  *((_QWORD *)a1 + 2) = a1 + 10;
  *((_QWORD *)a1 + 8) = a1 + 22;
  *a1 = 65793;
  a1[6] = 2;
  a1[18] = 4;
  *((_BYTE *)a1 + 84) = 1;
  return a1;
}
