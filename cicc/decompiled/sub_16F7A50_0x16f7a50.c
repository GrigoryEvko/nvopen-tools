// Function: sub_16F7A50
// Address: 0x16f7a50
//
__int64 __fastcall sub_16F7A50(__int64 *a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rcx
  __int64 result; // rax
  __int64 v7; // r15
  __int64 *v8; // r14
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // r12
  unsigned __int64 v12; // rax
  __int64 v13; // r13
  __int64 v14; // rax
  const char *v15; // [rsp-58h] [rbp-58h] BYREF
  char v16; // [rsp-48h] [rbp-48h]
  char v17; // [rsp-47h] [rbp-47h]

  v5 = a1[29];
  result = 3LL * *((unsigned int *)a1 + 60);
  if ( result )
  {
    v7 = a1[29];
    v8 = a1;
    do
    {
      while ( 1 )
      {
        v11 = v7 + 24;
        if ( *(_DWORD *)(v7 + 12) != *((_DWORD *)v8 + 16)
          || (unsigned int)(*(_DWORD *)(v7 + 8) + 1024) < *((_DWORD *)v8 + 15) )
        {
          break;
        }
        v7 += 24;
        result = v5 + 24LL * *((unsigned int *)v8 + 60);
        if ( v11 == result )
          return result;
      }
      if ( *(_BYTE *)(v7 + 20) )
      {
        v17 = 1;
        v12 = v8[6];
        v15 = "Could not find expected : for simple key";
        v16 = 3;
        if ( v8[5] >= v12 )
          v8[5] = v12 - 1;
        v13 = v8[43];
        if ( v13 )
        {
          v14 = sub_2241E50(a1, a2, a3, v5, a5);
          *(_DWORD *)v13 = 22;
          *(_QWORD *)(v13 + 8) = v14;
        }
        if ( !*((_BYTE *)v8 + 74) )
        {
          a1 = (__int64 *)*v8;
          a2 = v8[5];
          sub_16D14E0((__int64 *)*v8, a2, 0, (__int64)&v15, 0, 0, 0, 0, *((_BYTE *)v8 + 75));
        }
        *((_BYTE *)v8 + 74) = 1;
        v5 = v8[29];
      }
      v9 = *((unsigned int *)v8 + 60);
      a3 = v5 + 24 * v9;
      if ( a3 != v11 )
      {
        a2 = v7 + 24;
        a1 = (__int64 *)v7;
        memmove((void *)v7, (const void *)(v7 + 24), a3 - v11);
        LODWORD(v9) = *((_DWORD *)v8 + 60);
        v5 = v8[29];
      }
      v10 = (unsigned int)(v9 - 1);
      *((_DWORD *)v8 + 60) = v10;
      result = v5 + 24 * v10;
    }
    while ( v7 != result );
  }
  return result;
}
