// Function: sub_2EB4750
// Address: 0x2eb4750
//
__int64 __fastcall sub_2EB4750(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // r13d
  __int64 v7; // rax
  __int64 *v9; // r13
  __int64 v10; // rax
  __int64 *v11; // rdx
  __int64 *v13; // r14
  __int64 v14; // rcx
  __int64 v15; // rsi
  __int64 *v16; // rax
  __int64 *v17; // rbx
  __int64 v18; // r13
  __int64 v19; // rsi
  __int64 *v20; // rax
  __int64 *v21; // rdx
  __int64 v22; // [rsp+0h] [rbp-60h] BYREF
  __int64 *v23; // [rsp+8h] [rbp-58h]
  __int64 v24; // [rsp+10h] [rbp-50h]
  int v25; // [rsp+18h] [rbp-48h]
  unsigned __int8 v26; // [rsp+1Ch] [rbp-44h]
  _BYTE v27[64]; // [rsp+20h] [rbp-40h] BYREF

  v6 = 0;
  v7 = *(unsigned int *)(a1 + 8);
  if ( v7 != *(_DWORD *)(a2 + 8) )
    return v6;
  v9 = *(__int64 **)a1;
  v10 = v7;
  v11 = (__int64 *)v27;
  v26 = 1;
  v22 = 0;
  v13 = &v9[v10];
  v23 = (__int64 *)v27;
  v24 = 4;
  v25 = 0;
  if ( v9 != &v9[v10] )
  {
    v14 = 1;
    while ( 1 )
    {
      while ( 1 )
      {
        v15 = *v9;
        if ( (_BYTE)v14 )
          break;
LABEL_21:
        ++v9;
        sub_C8CC70((__int64)&v22, v15, (__int64)v11, v14, a5, a6);
        v14 = v26;
        if ( v13 == v9 )
          goto LABEL_11;
      }
      v16 = v23;
      v11 = &v23[HIDWORD(v24)];
      if ( v23 == v11 )
      {
LABEL_27:
        if ( HIDWORD(v24) >= (unsigned int)v24 )
          goto LABEL_21;
        ++v9;
        ++HIDWORD(v24);
        *v11 = v15;
        v14 = v26;
        ++v22;
        if ( v13 == v9 )
          goto LABEL_11;
      }
      else
      {
        while ( v15 != *v16 )
        {
          if ( v11 == ++v16 )
            goto LABEL_27;
        }
        if ( v13 == ++v9 )
        {
LABEL_11:
          v17 = *(__int64 **)a2;
          v18 = *(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8);
          if ( v18 != *(_QWORD *)a2 )
            goto LABEL_12;
LABEL_18:
          v6 = 1;
LABEL_19:
          if ( !(_BYTE)v14 )
          {
            _libc_free((unsigned __int64)v23);
            return v6;
          }
          return v6;
        }
      }
    }
  }
  v17 = *(__int64 **)a2;
  v18 = *(_QWORD *)a2 + v10 * 8;
  if ( v18 == *(_QWORD *)a2 )
    return 1;
  LOBYTE(v14) = 1;
  while ( 1 )
  {
LABEL_12:
    while ( 1 )
    {
      v19 = *v17;
      if ( (_BYTE)v14 )
        break;
      if ( !sub_C8CA60((__int64)&v22, v19) )
      {
        LOBYTE(v14) = v26;
        v6 = 0;
        goto LABEL_19;
      }
      ++v17;
      LOBYTE(v14) = v26;
      if ( (__int64 *)v18 == v17 )
        goto LABEL_18;
    }
    v20 = v23;
    v21 = &v23[HIDWORD(v24)];
    if ( v23 == v21 )
      return 0;
    while ( v19 != *v20 )
    {
      if ( v21 == ++v20 )
        return 0;
    }
    if ( (__int64 *)v18 == ++v17 )
      goto LABEL_18;
  }
}
