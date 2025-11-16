// Function: sub_2574030
// Address: 0x2574030
//
bool __fastcall sub_2574030(
        unsigned __int64 *a1,
        unsigned __int64 *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 *a8,
        __int64 a9,
        __int64 a10,
        _BYTE *a11)
{
  unsigned __int64 *v11; // r12
  unsigned __int64 *v13; // r14
  __int64 v14; // rdi
  __int64 v15; // rax
  unsigned __int64 v16; // rcx
  __int64 v17; // rsi
  unsigned int v18; // edx
  __int64 *v19; // rdi
  __int64 v20; // r8
  int v22; // edi
  int v23; // r10d
  unsigned __int64 v24[7]; // [rsp+28h] [rbp-38h] BYREF

  v11 = a1;
  if ( a2 != a1 )
  {
    v13 = a1;
    while ( 1 )
    {
      v15 = *(unsigned int *)(a7 + 184);
      v16 = v13[3];
      v17 = *(_QWORD *)(a7 + 168);
      if ( (_DWORD)v15 )
      {
        v18 = (v15 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
        v19 = (__int64 *)(v17 + 8LL * v18);
        v20 = *v19;
        if ( v16 == *v19 )
        {
LABEL_8:
          if ( v19 != (__int64 *)(v17 + 8 * v15) )
          {
            v14 = *a8;
            if ( *a8 )
            {
              v24[0] = v13[3];
              sub_2573E40(v14, (__int64 *)v24);
            }
            goto LABEL_5;
          }
        }
        else
        {
          v22 = 1;
          while ( v20 != -4096 )
          {
            v23 = v22 + 1;
            v18 = (v15 - 1) & (v22 + v18);
            v19 = (__int64 *)(v17 + 8LL * v18);
            v20 = *v19;
            if ( v16 == *v19 )
              goto LABEL_8;
            v22 = v23;
          }
        }
      }
      if ( !(unsigned __int8)sub_2522C50(a9, v13, a10, 0, a11, 0, 1) )
      {
LABEL_10:
        v11 = v13;
        return a2 == v11;
      }
LABEL_5:
      v13 = (unsigned __int64 *)v13[1];
      if ( a2 == v13 )
        goto LABEL_10;
    }
  }
  return a2 == v11;
}
