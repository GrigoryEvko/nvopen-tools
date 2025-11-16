// Function: sub_CA81D0
// Address: 0xca81d0
//
__int64 __fastcall sub_CA81D0(__int64 *a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rcx
  __int64 result; // rax
  __int64 v7; // r15
  __int64 *v8; // r14
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // r12
  __int64 v13; // rax
  unsigned __int64 v14; // r13
  unsigned __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // [rsp-70h] [rbp-70h]
  const char *v19; // [rsp-68h] [rbp-68h] BYREF
  char v20; // [rsp-48h] [rbp-48h]
  char v21; // [rsp-47h] [rbp-47h]

  v5 = a1[28];
  result = 3LL * *((unsigned int *)a1 + 58);
  if ( result )
  {
    v7 = a1[28];
    v8 = a1;
    do
    {
      while ( 1 )
      {
        v12 = v7 + 24;
        if ( *(_DWORD *)(v7 + 12) != *((_DWORD *)v8 + 16)
          || (unsigned int)(*(_DWORD *)(v7 + 8) + 1024) < *((_DWORD *)v8 + 15) )
        {
          break;
        }
        v7 += 24;
        result = v5 + 24LL * *((unsigned int *)v8 + 58);
        if ( v12 == result )
          return result;
      }
      if ( *(_BYTE *)(v7 + 20) )
      {
        v13 = *(_QWORD *)v7;
        v19 = "Could not find expected : for simple key";
        v14 = *(_QWORD *)(v13 + 24);
        v15 = v8[6];
        v21 = 1;
        v20 = 3;
        if ( v14 >= v15 )
          v14 = v15 - 1;
        v16 = v8[42];
        v18 = v16;
        if ( v16 )
        {
          v17 = sub_2241E50(a1, a2, v16, v5, a5);
          *(_DWORD *)v18 = 22;
          *(_QWORD *)(v18 + 8) = v17;
        }
        if ( !*((_BYTE *)v8 + 75) )
        {
          a1 = (__int64 *)*v8;
          a2 = v14;
          sub_C91CB0((__int64 *)*v8, v14, 0, (__int64)&v19, 0, 0, 0, 0, *((_BYTE *)v8 + 76));
        }
        *((_BYTE *)v8 + 75) = 1;
        v5 = v8[28];
      }
      v9 = *((unsigned int *)v8 + 58);
      v10 = v5 + 24 * v9;
      if ( v10 != v12 )
      {
        a2 = v7 + 24;
        a1 = (__int64 *)v7;
        memmove((void *)v7, (const void *)(v7 + 24), v10 - v12);
        LODWORD(v9) = *((_DWORD *)v8 + 58);
        v5 = v8[28];
      }
      v11 = (unsigned int)(v9 - 1);
      *((_DWORD *)v8 + 58) = v11;
      result = v5 + 24 * v11;
    }
    while ( v7 != result );
  }
  return result;
}
