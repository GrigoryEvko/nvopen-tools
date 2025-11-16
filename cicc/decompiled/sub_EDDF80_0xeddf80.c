// Function: sub_EDDF80
// Address: 0xeddf80
//
__int64 *__fastcall sub_EDDF80(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rax
  __int64 v8; // rcx
  __int64 v9; // r15
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // rdx
  __int64 v13; // rdx
  __int64 v14; // r8
  __int64 v15; // rcx
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // rsi
  _QWORD *v21; // rax
  _QWORD *v22; // r14
  _QWORD *v23; // rdi
  _QWORD *v24; // rdi
  _QWORD *v25; // rdi
  char v26; // al
  __int64 v27; // [rsp+18h] [rbp-338h]
  __int64 v28; // [rsp+20h] [rbp-330h]
  __int64 v29; // [rsp+28h] [rbp-328h] BYREF
  _QWORD v30[2]; // [rsp+30h] [rbp-320h] BYREF
  char v31; // [rsp+40h] [rbp-310h] BYREF
  char v32; // [rsp+120h] [rbp-230h]
  __int64 v33; // [rsp+130h] [rbp-220h] BYREF
  _QWORD v34[2]; // [rsp+138h] [rbp-218h] BYREF
  _BYTE v35[224]; // [rsp+148h] [rbp-208h] BYREF
  _BYTE *v36; // [rsp+228h] [rbp-128h]
  __int64 v37; // [rsp+230h] [rbp-120h]
  _BYTE v38[168]; // [rsp+238h] [rbp-118h] BYREF
  _BYTE *v39; // [rsp+2E0h] [rbp-70h]
  __int64 v40; // [rsp+2E8h] [rbp-68h]
  _BYTE v41[96]; // [rsp+2F0h] [rbp-60h] BYREF

  v4 = a4;
  v29 = a4;
  v8 = a4 + 8;
  v9 = *(_QWORD *)(v8 - 8);
  v29 = v8;
  v10 = *(_QWORD *)(v4 + 8);
  v11 = v4 + 16;
  v4 += 24;
  v29 = v11;
  v28 = v10;
  v12 = *(_QWORD *)(v4 - 8);
  v29 = v4;
  v27 = v12;
  sub_C16C80((__int64)v30, (__int64)&v29);
  v15 = v32 & 1;
  v32 = (2 * v15) | v32 & 0xFD;
  if ( (_BYTE)v15 )
  {
    *a1 = v30[0] | 1LL;
  }
  else
  {
    sub_ED6600(a2 + 8, (__int64)v30, v13, v15, v14, a2 + 8);
    v33 = 3;
    *(_QWORD *)(a2 + 272) = v29;
    *(_QWORD *)(a2 + 280) = a3 + v9;
    *(_DWORD *)(a2 + 288) = (unsigned __int64)(v28 - v9) >> 2;
    v34[0] = v35;
    v34[1] = 0x1C00000000LL;
    if ( *(_DWORD *)(a2 + 16) )
      sub_ED6600((__int64)v34, a2 + 8, v17, v18, v19, a2 + 8);
    v36 = v38;
    v20 = a3 + v28;
    v37 = 0x100000000LL;
    v39 = v41;
    v40 = 0x600000000LL;
    v21 = sub_ED8DB0((__int64 *)(a3 + v27), a3 + v28, a3, (__int64)&v33);
    v22 = *(_QWORD **)(a2 + 248);
    *(_QWORD *)(a2 + 248) = v21;
    if ( v22 )
    {
      v23 = (_QWORD *)v22[58];
      if ( v23 != v22 + 60 )
        _libc_free(v23, v20);
      v24 = (_QWORD *)v22[35];
      if ( v24 != v22 + 37 )
        _libc_free(v24, v20);
      v25 = (_QWORD *)v22[5];
      if ( v25 != v22 + 7 )
        _libc_free(v25, v20);
      v20 = 536;
      j_j___libc_free_0(v22, 536);
    }
    if ( v39 != v41 )
      _libc_free(v39, v20);
    if ( v36 != v38 )
      _libc_free(v36, v20);
    if ( (_BYTE *)v34[0] != v35 )
      _libc_free(v34[0], v20);
    v26 = v32;
    *a1 = 1;
    if ( (v26 & 2) != 0 )
      sub_EDDB10(v30, v20);
    if ( (v26 & 1) != 0 )
    {
      if ( v30[0] )
        (*(void (**)(void))(*(_QWORD *)v30[0] + 8LL))();
    }
    else if ( (char *)v30[0] != &v31 )
    {
      _libc_free(v30[0], v20);
    }
  }
  return a1;
}
