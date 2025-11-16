// Function: sub_1C2E890
// Address: 0x1c2e890
//
__int64 __fastcall sub_1C2E890(__int64 a1)
{
  unsigned __int8 v2; // al
  unsigned int v3; // r12d
  __int64 v4; // rdi
  _DWORD *v5; // rsi
  _BYTE *v6; // r8
  int v8; // [rsp+Ch] [rbp-74h] BYREF
  _DWORD *v9; // [rsp+10h] [rbp-70h] BYREF
  __int64 v10; // [rsp+18h] [rbp-68h]
  _BYTE v11[96]; // [rsp+20h] [rbp-60h] BYREF

  v2 = *(_BYTE *)(a1 + 16);
  if ( v2 <= 3u )
  {
    v3 = sub_1C2E690(a1, "sampler", 7u, &v9);
    if ( (_BYTE)v3 )
      return v3;
    v2 = *(_BYTE *)(a1 + 16);
  }
  v3 = 0;
  if ( v2 != 17 )
    return v3;
  v4 = *(_QWORD *)(a1 + 24);
  v9 = v11;
  v10 = 0x1000000000LL;
  v3 = sub_1C2E2E0(v4, "sampler", 7u, (__int64)&v9);
  if ( (_BYTE)v3 )
  {
    v8 = *(_DWORD *)(a1 + 32);
    v5 = &v9[(unsigned int)v10];
    if ( v5 != sub_1C2E030(v9, (__int64)v5, &v8) )
    {
      if ( v6 != v11 )
        _libc_free((unsigned __int64)v6);
      return v3;
    }
  }
  else
  {
    v6 = v9;
  }
  if ( v6 != v11 )
    _libc_free((unsigned __int64)v6);
  return 0;
}
