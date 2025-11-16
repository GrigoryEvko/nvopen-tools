// Function: sub_2D47210
// Address: 0x2d47210
//
__int64 __fastcall sub_2D47210(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rcx
  __int64 v5; // rax
  __int64 v6; // rbx
  __int64 v7; // r12
  __int64 v8; // r15
  __int64 v9; // rax
  char v10; // bl
  _QWORD *v11; // rax
  __int64 v12; // r9
  __int64 v13; // r12
  char *v14; // rbx
  char *v15; // r13
  __int64 v16; // rdx
  unsigned int v17; // esi
  unsigned __int64 v18; // rcx
  __int16 v19; // dx
  __int16 v20; // ax
  __int16 v21; // ax
  int v23; // r12d
  char *v24; // rbx
  char *v25; // r12
  __int64 v26; // rdx
  unsigned int v27; // esi
  __int64 v28; // [rsp+8h] [rbp-1A8h]
  _BYTE v29[32]; // [rsp+10h] [rbp-1A0h] BYREF
  __int16 v30; // [rsp+30h] [rbp-180h]
  _BYTE v31[32]; // [rsp+40h] [rbp-170h] BYREF
  __int16 v32; // [rsp+60h] [rbp-150h]
  char *v33; // [rsp+70h] [rbp-140h] BYREF
  int v34; // [rsp+78h] [rbp-138h]
  char v35; // [rsp+80h] [rbp-130h] BYREF
  __int64 v36; // [rsp+A0h] [rbp-110h]
  __int64 v37; // [rsp+A8h] [rbp-108h]
  __int64 v38; // [rsp+B0h] [rbp-100h]
  __int64 v39; // [rsp+C0h] [rbp-F0h]
  __int64 v40; // [rsp+C8h] [rbp-E8h]
  __int64 v41; // [rsp+D0h] [rbp-E0h]
  int v42; // [rsp+D8h] [rbp-D8h]
  void *v43; // [rsp+F0h] [rbp-C0h]
  void *v44; // [rsp+F8h] [rbp-B8h]
  _QWORD v45[12]; // [rsp+150h] [rbp-60h] BYREF

  sub_2D46B10((__int64)&v33, a2, *(_QWORD *)(a1 + 8));
  v3 = sub_B43CA0(a2);
  v5 = sub_2D43EB0(*(_QWORD *)a1, *(__int64 **)(*(_QWORD *)(a2 - 64) + 8LL), v3 + 312, v4);
  v6 = *(_QWORD *)(a2 - 64);
  v30 = 257;
  if ( v5 == *(_QWORD *)(v6 + 8) )
  {
    v8 = v6;
  }
  else
  {
    v7 = v5;
    v8 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v39 + 120LL))(v39, 49, v6, v5);
    if ( !v8 )
    {
      v32 = 257;
      v8 = sub_B51D30(49, v6, v7, (__int64)v31, 0, 0);
      if ( (unsigned __int8)sub_920620(v8) )
      {
        v23 = v42;
        if ( v41 )
          sub_B99FD0(v8, 3u, v41);
        sub_B45150(v8, v23);
      }
      (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)v40 + 16LL))(
        v40,
        v8,
        v29,
        v37,
        v38);
      v24 = v33;
      v25 = &v33[16 * v34];
      if ( v33 != v25 )
      {
        do
        {
          v26 = *((_QWORD *)v24 + 1);
          v27 = *(_DWORD *)v24;
          v24 += 16;
          sub_B99FD0(v8, v27, v26);
        }
        while ( v25 != v24 );
      }
    }
  }
  v28 = *(_QWORD *)(a2 - 32);
  v9 = sub_AA4E30(v36);
  v10 = sub_AE5020(v9, *(_QWORD *)(v8 + 8));
  v32 = 257;
  v11 = sub_BD2C40(80, unk_3F10A10);
  v13 = (__int64)v11;
  if ( v11 )
    sub_B4D3C0((__int64)v11, v8, v28, 0, v10, v12, 0, 0);
  (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)v40 + 16LL))(v40, v13, v31, v37, v38);
  v14 = v33;
  v15 = &v33[16 * v34];
  if ( v33 != v15 )
  {
    do
    {
      v16 = *((_QWORD *)v14 + 1);
      v17 = *(_DWORD *)v14;
      v14 += 16;
      sub_B99FD0(v13, v17, v16);
    }
    while ( v15 != v14 );
  }
  _BitScanReverse64(&v18, 1LL << (*(_WORD *)(a2 + 2) >> 1));
  v19 = (2 * (63 - (v18 ^ 0x3F))) | *(_WORD *)(v13 + 2) & 0xFF81;
  *(_WORD *)(v13 + 2) = v19;
  v20 = v19 & 0xFFFE | *(_WORD *)(a2 + 2) & 1;
  *(_WORD *)(v13 + 2) = v20;
  v21 = *(_WORD *)(a2 + 2) & 0x380 | v20 & 0xFC7F;
  *(_BYTE *)(v13 + 72) = *(_BYTE *)(a2 + 72);
  *(_WORD *)(v13 + 2) = v21;
  sub_B43D60((_QWORD *)a2);
  sub_B32BF0(v45);
  v43 = &unk_49E5698;
  v44 = &unk_49D94D0;
  nullsub_63();
  nullsub_63();
  if ( v33 != &v35 )
    _libc_free((unsigned __int64)v33);
  return v13;
}
