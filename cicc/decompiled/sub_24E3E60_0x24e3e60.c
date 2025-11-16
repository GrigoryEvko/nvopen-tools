// Function: sub_24E3E60
// Address: 0x24e3e60
//
__int64 __fastcall sub_24E3E60(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  unsigned __int8 *v3; // rax
  __int64 v4; // r8
  __int64 v5; // r9
  unsigned __int64 v7[2]; // [rsp+20h] [rbp-2E0h] BYREF
  _BYTE v8[64]; // [rsp+30h] [rbp-2D0h] BYREF
  _BYTE *v9; // [rsp+70h] [rbp-290h]
  __int64 v10; // [rsp+78h] [rbp-288h]
  _BYTE v11[64]; // [rsp+80h] [rbp-280h] BYREF
  _BYTE *v12; // [rsp+C0h] [rbp-240h]
  __int64 v13; // [rsp+C8h] [rbp-238h]
  _BYTE v14[64]; // [rsp+D0h] [rbp-230h] BYREF
  _BYTE *v15; // [rsp+110h] [rbp-1F0h]
  __int64 v16; // [rsp+118h] [rbp-1E8h]
  _BYTE v17[64]; // [rsp+120h] [rbp-1E0h] BYREF
  _BYTE *v18; // [rsp+160h] [rbp-1A0h]
  __int64 v19; // [rsp+168h] [rbp-198h]
  _BYTE v20[64]; // [rsp+170h] [rbp-190h] BYREF
  __int64 v21; // [rsp+1B0h] [rbp-150h]
  char *v22; // [rsp+1B8h] [rbp-148h]
  __int64 v23; // [rsp+1C0h] [rbp-140h]
  int v24; // [rsp+1C8h] [rbp-138h]
  char v25; // [rsp+1CCh] [rbp-134h]
  char v26; // [rsp+1D0h] [rbp-130h] BYREF

  v2 = sub_C996C0("CollectCommonDebugInfo", 22, 0, 0);
  v18 = v20;
  v7[0] = (unsigned __int64)v8;
  v7[1] = 0x800000000LL;
  v10 = 0x800000000LL;
  v13 = 0x800000000LL;
  v16 = 0x800000000LL;
  v19 = 0x800000000LL;
  v9 = v11;
  v12 = v14;
  v15 = v17;
  v21 = 0;
  v22 = &v26;
  v23 = 32;
  v24 = 0;
  v25 = 1;
  v3 = (unsigned __int8 *)sub_F459D0(a2, 0, (__int64)v7);
  sub_F45A20(a1, 0, (__int64)v7, v3, v4, v5);
  if ( !v25 )
    _libc_free((unsigned __int64)v22);
  if ( v18 != v20 )
    _libc_free((unsigned __int64)v18);
  if ( v15 != v17 )
    _libc_free((unsigned __int64)v15);
  if ( v12 != v14 )
    _libc_free((unsigned __int64)v12);
  if ( v9 != v11 )
    _libc_free((unsigned __int64)v9);
  if ( (_BYTE *)v7[0] != v8 )
    _libc_free(v7[0]);
  if ( v2 )
    sub_C9AF60(v2);
  return a1;
}
