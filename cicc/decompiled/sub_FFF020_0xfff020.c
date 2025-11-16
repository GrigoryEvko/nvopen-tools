// Function: sub_FFF020
// Address: 0xfff020
//
__int64 __fastcall sub_FFF020(__int64 a1, _BYTE *a2, unsigned __int8 **a3, unsigned __int64 a4, __int64 a5)
{
  __int64 v5; // r13
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 *v11; // r10
  __int64 i; // r8
  unsigned __int8 *v13; // r12
  unsigned __int8 v14; // al
  __int64 v15; // rax
  unsigned __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // r12
  __int64 v20; // [rsp+8h] [rbp-78h]
  __int64 *v21; // [rsp+10h] [rbp-70h]
  __int64 *v22; // [rsp+18h] [rbp-68h]
  __int64 *v23; // [rsp+20h] [rbp-60h] BYREF
  __int64 v24; // [rsp+28h] [rbp-58h]
  _BYTE v25[80]; // [rsp+30h] [rbp-50h] BYREF

  if ( *a2 )
    return 0;
  v5 = (__int64)a2;
  if ( !sub_971E80(a1, (__int64)a2) )
    return 0;
  v11 = (__int64 *)v25;
  v23 = (__int64 *)v25;
  v24 = 0x400000000LL;
  if ( a4 > 4 )
  {
    a2 = v25;
    sub_C8D5F0((__int64)&v23, v25, a4, 8u, v9, v10);
    v11 = (__int64 *)v25;
  }
  for ( i = (__int64)&a3[a4]; (unsigned __int8 **)i != a3; ++a3 )
  {
    v13 = *a3;
    v14 = **a3;
    if ( v14 > 0x15u )
    {
      if ( v14 != 24 )
      {
        v18 = 0;
        goto LABEL_14;
      }
    }
    else
    {
      v15 = (unsigned int)v24;
      v16 = (unsigned int)v24 + 1LL;
      if ( v16 > HIDWORD(v24) )
      {
        a2 = v11;
        v20 = i;
        v21 = v11;
        sub_C8D5F0((__int64)&v23, v11, v16, 8u, i, v10);
        v15 = (unsigned int)v24;
        i = v20;
        v11 = v21;
      }
      v23[v15] = (__int64)v13;
      LODWORD(v24) = v24 + 1;
    }
  }
  a2 = (_BYTE *)v5;
  v22 = v11;
  v17 = sub_97A150(a1, v5, v23, (unsigned int)v24, *(__int64 **)(a5 + 8), 1);
  v11 = v22;
  v18 = v17;
LABEL_14:
  if ( v23 != v11 )
    _libc_free(v23, a2);
  return v18;
}
