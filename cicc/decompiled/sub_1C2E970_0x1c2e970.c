// Function: sub_1C2E970
// Address: 0x1c2e970
//
__int64 __fastcall sub_1C2E970(__int64 a1)
{
  unsigned int v1; // r12d
  __int64 v2; // rax
  __int64 v3; // rdi
  __int64 v4; // rbx
  _BYTE *v5; // r8
  _DWORD *v7; // rsi
  int v8; // [rsp+Ch] [rbp-74h] BYREF
  _DWORD *v9; // [rsp+10h] [rbp-70h] BYREF
  __int64 v10; // [rsp+18h] [rbp-68h]
  _BYTE v11[96]; // [rsp+20h] [rbp-60h] BYREF

  v1 = 0;
  v2 = sub_1649C60(a1);
  if ( *(_BYTE *)(v2 + 16) != 17 )
    return v1;
  v3 = *(_QWORD *)(v2 + 24);
  v4 = v2;
  v9 = v11;
  v10 = 0x1000000000LL;
  v1 = sub_1C2E2E0(v3, "rdoimage", 8u, (__int64)&v9);
  if ( !(_BYTE)v1 )
  {
    v5 = v9;
LABEL_4:
    if ( v5 != v11 )
      _libc_free((unsigned __int64)v5);
    return 0;
  }
  v8 = *(_DWORD *)(v4 + 32);
  v7 = &v9[(unsigned int)v10];
  if ( v7 == sub_1C2E030(v9, (__int64)v7, &v8) )
    goto LABEL_4;
  if ( v5 != v11 )
    _libc_free((unsigned __int64)v5);
  return v1;
}
