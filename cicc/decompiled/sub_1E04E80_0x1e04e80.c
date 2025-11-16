// Function: sub_1E04E80
// Address: 0x1e04e80
//
__int64 __fastcall sub_1E04E80(__int64 a1, __int64 a2)
{
  unsigned int v2; // r8d
  __int64 **v3; // r15
  __int64 **v4; // r13
  __int64 **v5; // r14
  __int64 **v6; // rdx
  __int64 *v8; // rax
  __int64 **v9; // r14
  __int64 *v10; // rdi
  __int64 v11; // r8
  __int64 *v12; // r9
  __int64 *v13; // rdx
  __int64 *v14; // rsi
  __int64 *v15; // rbx
  bool v16; // zf
  __int64 v17; // r13
  __int64 *v19; // rdx
  unsigned __int8 v20; // [rsp+Fh] [rbp-81h]
  __int64 v21; // [rsp+10h] [rbp-80h] BYREF
  __int64 *v22; // [rsp+18h] [rbp-78h]
  __int64 *v23; // [rsp+20h] [rbp-70h]
  __int64 v24; // [rsp+28h] [rbp-68h]
  int v25; // [rsp+30h] [rbp-60h]
  _BYTE v26[88]; // [rsp+38h] [rbp-58h] BYREF

  v2 = 1;
  v3 = *(__int64 ***)(a1 + 32);
  v4 = *(__int64 ***)(a2 + 32);
  v5 = *(__int64 ***)(a1 + 24);
  v6 = *(__int64 ***)(a2 + 24);
  if ( (char *)v3 - (char *)v5 != (char *)v4 - (char *)v6 || *(_DWORD *)(a1 + 16) != *(_DWORD *)(a2 + 16) )
    return v2;
  v8 = (__int64 *)v26;
  v21 = 0;
  v22 = (__int64 *)v26;
  v23 = (__int64 *)v26;
  v24 = 4;
  v25 = 0;
  if ( v6 == v4 )
  {
    v10 = (__int64 *)v26;
    if ( v5 == v3 )
      return 0;
    while ( 1 )
    {
LABEL_23:
      v17 = **v5;
      if ( v10 == v8 )
      {
        v15 = &v8[HIDWORD(v24)];
        if ( v8 == v15 )
        {
          v10 = v23;
          v19 = v8;
        }
        else
        {
          do
          {
            if ( v17 == *v8 )
              break;
            ++v8;
          }
          while ( v15 != v8 );
          v10 = v23;
          v19 = v15;
        }
LABEL_31:
        while ( v19 != v8 )
        {
          if ( (unsigned __int64)*v8 < 0xFFFFFFFFFFFFFFFELL )
            goto LABEL_21;
          ++v8;
        }
        v16 = v8 == v15;
        v8 = v22;
        if ( v16 )
          goto LABEL_33;
      }
      else
      {
        v15 = &v10[(unsigned int)v24];
        v8 = sub_16CC9F0((__int64)&v21, **v5);
        if ( v17 == *v8 )
        {
          v10 = v23;
          if ( v23 == v22 )
            v19 = &v23[HIDWORD(v24)];
          else
            v19 = &v23[(unsigned int)v24];
          goto LABEL_31;
        }
        v10 = v23;
        if ( v23 == v22 )
        {
          v8 = &v23[HIDWORD(v24)];
          v19 = v8;
          goto LABEL_31;
        }
        v8 = &v23[(unsigned int)v24];
LABEL_21:
        v16 = v8 == v15;
        v8 = v22;
        if ( v16 )
        {
LABEL_33:
          v2 = 1;
          goto LABEL_34;
        }
      }
      if ( v3 == ++v5 )
        goto LABEL_42;
    }
  }
  v9 = v6;
  v10 = (__int64 *)v26;
  do
  {
LABEL_7:
    v11 = **v9;
    if ( v8 != v10 )
    {
LABEL_5:
      sub_16CCBA0((__int64)&v21, **v9);
      v10 = v23;
      v8 = v22;
      goto LABEL_6;
    }
    v12 = &v8[HIDWORD(v24)];
    if ( v12 == v8 )
    {
LABEL_40:
      if ( HIDWORD(v24) >= (unsigned int)v24 )
        goto LABEL_5;
      ++HIDWORD(v24);
      *v12 = v11;
      v8 = v22;
      ++v21;
      v10 = v23;
    }
    else
    {
      v13 = v8;
      v14 = 0;
      while ( v11 != *v13 )
      {
        if ( *v13 == -2 )
          v14 = v13;
        if ( v12 == ++v13 )
        {
          if ( !v14 )
            goto LABEL_40;
          ++v9;
          *v14 = v11;
          v10 = v23;
          --v25;
          v8 = v22;
          ++v21;
          if ( v4 != v9 )
            goto LABEL_7;
          goto LABEL_16;
        }
      }
    }
LABEL_6:
    ++v9;
  }
  while ( v4 != v9 );
LABEL_16:
  v3 = *(__int64 ***)(a1 + 32);
  v5 = *(__int64 ***)(a1 + 24);
  if ( v3 != v5 )
    goto LABEL_23;
LABEL_42:
  v2 = 0;
LABEL_34:
  if ( v8 != v10 )
  {
    v20 = v2;
    _libc_free((unsigned __int64)v10);
    return v20;
  }
  return v2;
}
