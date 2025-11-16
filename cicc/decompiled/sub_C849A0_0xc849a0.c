// Function: sub_C849A0
// Address: 0xc849a0
//
__int64 __fastcall sub_C849A0(__int64 a1)
{
  __int64 v1; // rsi
  unsigned __int8 *v2; // rax
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // r8
  unsigned int v6; // r12d
  unsigned int v8; // eax
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  _QWORD v12[4]; // [rsp+10h] [rbp-F0h] BYREF
  __int16 v13; // [rsp+30h] [rbp-D0h]
  const char *v14; // [rsp+40h] [rbp-C0h] BYREF
  __int64 v15; // [rsp+48h] [rbp-B8h]
  __int64 v16; // [rsp+50h] [rbp-B0h]
  char v17; // [rsp+58h] [rbp-A8h] BYREF
  __int16 v18; // [rsp+60h] [rbp-A0h]

  v1 = 0;
  v2 = *(unsigned __int8 **)a1;
  v18 = 261;
  v14 = (const char *)v2;
  v15 = *(_QWORD *)(a1 + 8);
  if ( (unsigned __int8)sub_C81DB0(&v14, 0) )
  {
    sub_2241E40(&v14, 0, v3, v4, v5);
    return 0;
  }
  else
  {
    v15 = 0;
    v14 = &v17;
    v16 = 128;
    v8 = sub_C82800(&v14);
    if ( v8 )
    {
      v6 = v8;
    }
    else
    {
      v1 = a1;
      v6 = 0;
      v13 = 261;
      v12[0] = v14;
      v12[1] = v15;
      sub_C846B0((__int64)v12, (unsigned __int8 **)a1);
      sub_2241E40(v12, a1, v9, v10, v11);
    }
    if ( v14 != &v17 )
      _libc_free(v14, v1);
  }
  return v6;
}
