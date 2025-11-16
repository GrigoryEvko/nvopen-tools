// Function: sub_25107F0
// Address: 0x25107f0
//
__int64 __fastcall sub_25107F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  size_t v5; // r13
  const void *v6; // r15
  __int64 v7; // r9
  unsigned int v8; // r12d
  _BYTE *v10; // rdi
  __int64 v11; // [rsp+18h] [rbp-78h] BYREF
  _BYTE *v12; // [rsp+20h] [rbp-70h] BYREF
  size_t v13; // [rsp+28h] [rbp-68h]
  __int64 v14; // [rsp+30h] [rbp-60h]
  _BYTE v15[88]; // [rsp+38h] [rbp-58h] BYREF

  v5 = *(_QWORD *)(a2 + 8);
  v6 = *(const void **)a2;
  v11 = a3;
  v8 = sub_A73380(&v11, v6, v5);
  if ( !(_BYTE)v8 )
    return v8;
  v13 = 0;
  v12 = v15;
  v14 = 32;
  if ( v5 > 0x20 )
  {
    sub_C8D290((__int64)&v12, v15, v5, 1u, (__int64)&v12, v7);
    v10 = &v12[v13];
  }
  else
  {
    if ( !v5 )
      goto LABEL_5;
    v10 = v15;
  }
  memcpy(v10, v6, v5);
  v5 += v13;
LABEL_5:
  v13 = v5;
  sub_25104F0((_QWORD *)(a4 + 16), &v12);
  if ( v12 != v15 )
    _libc_free((unsigned __int64)v12);
  return v8;
}
