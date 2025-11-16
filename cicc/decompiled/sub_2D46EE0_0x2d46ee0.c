// Function: sub_2D46EE0
// Address: 0x2d46ee0
//
__int64 __fastcall sub_2D46EE0(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rcx
  __int64 v5; // r12
  __int64 v6; // r14
  __int64 v7; // rax
  char v8; // bl
  _QWORD *v9; // rax
  __int64 v10; // r15
  char *v11; // rbx
  char *v12; // r12
  __int64 v13; // rdx
  unsigned int v14; // esi
  unsigned __int64 v15; // rcx
  __int16 v16; // dx
  __int16 v17; // ax
  __int16 v18; // ax
  __int64 v19; // r12
  __int64 v20; // r14
  int v22; // r12d
  char *v23; // rbx
  char *v24; // r12
  __int64 v25; // rdx
  unsigned int v26; // esi
  _BYTE v27[32]; // [rsp+10h] [rbp-1A0h] BYREF
  __int16 v28; // [rsp+30h] [rbp-180h]
  _BYTE v29[32]; // [rsp+40h] [rbp-170h] BYREF
  __int16 v30; // [rsp+60h] [rbp-150h]
  char *v31; // [rsp+70h] [rbp-140h] BYREF
  int v32; // [rsp+78h] [rbp-138h]
  char v33; // [rsp+80h] [rbp-130h] BYREF
  __int64 v34; // [rsp+A0h] [rbp-110h]
  __int64 v35; // [rsp+A8h] [rbp-108h]
  __int64 v36; // [rsp+B0h] [rbp-100h]
  __int64 v37; // [rsp+C0h] [rbp-F0h]
  __int64 v38; // [rsp+C8h] [rbp-E8h]
  __int64 v39; // [rsp+D0h] [rbp-E0h]
  int v40; // [rsp+D8h] [rbp-D8h]
  void *v41; // [rsp+F0h] [rbp-C0h]
  void *v42; // [rsp+F8h] [rbp-B8h]
  _QWORD v43[12]; // [rsp+150h] [rbp-60h] BYREF

  v3 = sub_B43CA0(a2);
  v5 = sub_2D43EB0(*(_QWORD *)a1, *(__int64 **)(a2 + 8), v3 + 312, v4);
  sub_2D46B10((__int64)&v31, a2, *(_QWORD *)(a1 + 8));
  v6 = *(_QWORD *)(a2 - 32);
  v28 = 257;
  v7 = sub_AA4E30(v34);
  v8 = sub_AE5020(v7, v5);
  v30 = 257;
  v9 = sub_BD2C40(80, 1u);
  v10 = (__int64)v9;
  if ( v9 )
    sub_B4D190((__int64)v9, v5, v6, (__int64)v29, 0, v8, 0, 0);
  (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)v38 + 16LL))(v38, v10, v27, v35, v36);
  v11 = v31;
  v12 = &v31[16 * v32];
  if ( v31 != v12 )
  {
    do
    {
      v13 = *((_QWORD *)v11 + 1);
      v14 = *(_DWORD *)v11;
      v11 += 16;
      sub_B99FD0(v10, v14, v13);
    }
    while ( v12 != v11 );
  }
  _BitScanReverse64(&v15, 1LL << (*(_WORD *)(a2 + 2) >> 1));
  v16 = (2 * (63 - (v15 ^ 0x3F))) | *(_WORD *)(v10 + 2) & 0xFF81;
  *(_WORD *)(v10 + 2) = v16;
  v17 = v16 & 0xFFFE | *(_WORD *)(a2 + 2) & 1;
  *(_WORD *)(v10 + 2) = v17;
  v18 = *(_WORD *)(a2 + 2) & 0x380 | v17 & 0xFC7F;
  *(_BYTE *)(v10 + 72) = *(_BYTE *)(a2 + 72);
  *(_WORD *)(v10 + 2) = v18;
  v19 = *(_QWORD *)(a2 + 8);
  v28 = 257;
  if ( v19 == *(_QWORD *)(v10 + 8) )
  {
    v20 = v10;
  }
  else
  {
    v20 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v37 + 120LL))(v37, 49, v10, v19);
    if ( !v20 )
    {
      v30 = 257;
      v20 = sub_B51D30(49, v10, v19, (__int64)v29, 0, 0);
      if ( (unsigned __int8)sub_920620(v20) )
      {
        v22 = v40;
        if ( v39 )
          sub_B99FD0(v20, 3u, v39);
        sub_B45150(v20, v22);
      }
      (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)v38 + 16LL))(
        v38,
        v20,
        v27,
        v35,
        v36);
      v23 = v31;
      v24 = &v31[16 * v32];
      if ( v31 != v24 )
      {
        do
        {
          v25 = *((_QWORD *)v23 + 1);
          v26 = *(_DWORD *)v23;
          v23 += 16;
          sub_B99FD0(v20, v26, v25);
        }
        while ( v24 != v23 );
      }
    }
  }
  sub_BD84D0(a2, v20);
  sub_B43D60((_QWORD *)a2);
  sub_B32BF0(v43);
  v41 = &unk_49E5698;
  v42 = &unk_49D94D0;
  nullsub_63();
  nullsub_63();
  if ( v31 != &v33 )
    _libc_free((unsigned __int64)v31);
  return v10;
}
