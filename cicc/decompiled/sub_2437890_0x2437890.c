// Function: sub_2437890
// Address: 0x2437890
//
void __fastcall sub_2437890(__int64 a1, __int64 a2, __int64 a3, unsigned __int64 a4, unsigned __int64 a5)
{
  unsigned __int64 v5; // r15
  unsigned __int64 v6; // r13
  __int64 v8; // rbx
  bool v9; // zf
  __int64 v10; // r10
  __int64 v11; // rdi
  __int64 (__fastcall *v12)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v13; // rax
  __int64 v14; // r14
  __int64 v15; // rdx
  _BYTE *v16; // rax
  __int64 v17; // rdi
  __int64 v18; // rax
  __int64 v19; // rdx
  unsigned __int64 v20; // rsi
  __int64 v21; // r15
  unsigned int *v22; // rbx
  __int64 v23; // rdx
  unsigned int v24; // esi
  char v25; // cl
  __int64 v26; // rdx
  __int64 v27; // r15
  _BYTE *v28; // rax
  __int64 v29; // rax
  unsigned __int64 v30; // rax
  __int64 v31; // r10
  char v32; // cl
  unsigned __int64 v33; // rax
  __int64 v34; // rsi
  __int64 v35; // r13
  __int64 v36; // rax
  char v37; // al
  _QWORD *v38; // rax
  __int64 v39; // r9
  __int64 v40; // r15
  __int64 v41; // r13
  unsigned int *v42; // rbx
  __int64 v43; // rdx
  unsigned int v44; // esi
  __int64 v45; // rdx
  _BYTE *v46; // rax
  __int64 v47; // r15
  __int64 v48; // rax
  char v49; // bl
  _QWORD *v50; // rax
  __int64 v51; // r9
  __int64 v52; // r13
  unsigned int *v53; // rbx
  __int64 v54; // r12
  __int64 v55; // rdx
  unsigned int v56; // esi
  __int64 v57; // rax
  __int64 v58; // rax
  __int64 v59; // rax
  __int64 v60; // [rsp-8h] [rbp-108h]
  __int64 v61; // [rsp+8h] [rbp-F8h]
  char v62; // [rsp+14h] [rbp-ECh]
  __int64 v63; // [rsp+18h] [rbp-E8h]
  __int64 v64; // [rsp+20h] [rbp-E0h]
  __int64 v65; // [rsp+20h] [rbp-E0h]
  __int64 **v67; // [rsp+30h] [rbp-D0h]
  __int64 v68; // [rsp+30h] [rbp-D0h]
  unsigned __int64 v69; // [rsp+38h] [rbp-C8h]
  _QWORD v70[4]; // [rsp+40h] [rbp-C0h] BYREF
  __int16 v71; // [rsp+60h] [rbp-A0h]
  _BYTE v72[32]; // [rsp+70h] [rbp-90h] BYREF
  __int16 v73; // [rsp+90h] [rbp-70h]
  _BYTE v74[32]; // [rsp+A0h] [rbp-60h] BYREF
  __int16 v75; // [rsp+C0h] [rbp-40h]

  v5 = a4;
  v6 = a5;
  v8 = a1;
  LOBYTE(a4) = *(_BYTE *)(a1 + 104);
  v69 = 1LL << a4;
  if ( 1LL << a4 )
  {
    _BitScanReverse64(&a4, 1LL << a4);
    v69 = -(__int64)(0x8000000000000000LL >> ((unsigned __int8)a4 ^ 0x3Fu))
        & ((0x8000000000000000LL >> ((unsigned __int8)a4 ^ 0x3Fu)) + a5 - 1);
  }
  v9 = *(_BYTE *)(a1 + 164) == 0;
  v10 = *(_QWORD *)(a1 + 136);
  v73 = 257;
  if ( v9 )
    v6 = v69;
  if ( v10 == *(_QWORD *)(v5 + 8) )
  {
    v14 = v5;
    goto LABEL_12;
  }
  v11 = *(_QWORD *)(a2 + 80);
  v12 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v11 + 120LL);
  if ( v12 != sub_920130 )
  {
    v68 = v10;
    v59 = v12(v11, 38u, (_BYTE *)v5, v10);
    v10 = v68;
    v14 = v59;
    goto LABEL_11;
  }
  if ( *(_BYTE *)v5 <= 0x15u )
  {
    v67 = (__int64 **)v10;
    if ( (unsigned __int8)sub_AC4810(0x26u) )
      v13 = sub_ADAB70(38, v5, v67, 0);
    else
      v13 = sub_AA93C0(0x26u, v5, (__int64)v67);
    v10 = (__int64)v67;
    v14 = v13;
LABEL_11:
    if ( v14 )
      goto LABEL_12;
  }
  v75 = 257;
  v14 = sub_B51D30(38, v5, v10, (__int64)v74, 0, 0);
  (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
    *(_QWORD *)(a2 + 88),
    v14,
    v72,
    *(_QWORD *)(a2 + 56),
    *(_QWORD *)(a2 + 64));
  if ( *(_QWORD *)a2 != *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8) )
  {
    v64 = v8;
    v21 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
    v22 = *(unsigned int **)a2;
    do
    {
      v23 = *((_QWORD *)v22 + 1);
      v24 = *v22;
      v22 += 4;
      sub_B99FD0(v14, v24, v23);
    }
    while ( (unsigned int *)v21 != v22 );
    v8 = v64;
    if ( *(_BYTE *)(v64 + 166) )
      goto LABEL_13;
    goto LABEL_19;
  }
LABEL_12:
  if ( *(_BYTE *)(v8 + 166) )
  {
LABEL_13:
    v15 = *(_QWORD *)(v8 + 128);
    v75 = 257;
    v73 = 257;
    v16 = sub_94BCF0((unsigned int **)a2, a3, v15, (__int64)v72);
    v17 = *(_QWORD *)(v8 + 120);
    v70[0] = v16;
    v70[1] = v14;
    v18 = sub_AD64C0(v17, v69, 0);
    v19 = *(_QWORD *)(v8 + 464);
    v20 = *(_QWORD *)(v8 + 456);
    v70[2] = v18;
    sub_921880((unsigned int **)a2, v20, v19, (int)v70, 3, (__int64)v74, 0);
    return;
  }
LABEL_19:
  v25 = *(_BYTE *)(v8 + 104);
  v26 = *(_QWORD *)(v8 + 120);
  v75 = 257;
  v27 = v6 >> v25;
  v28 = sub_94BCF0((unsigned int **)a2, a3, v26, (__int64)v74);
  v29 = sub_2435400(*(_BYTE *)(v8 + 160), *(_DWORD *)(v8 + 176), *(_QWORD *)(v8 + 184), (__int64 *)a2, (__int64)v28);
  v30 = sub_2436FF0(v8, v29, a2);
  v31 = v30;
  if ( v27 )
  {
    v65 = v30;
    v57 = sub_BCB2E0(*(_QWORD **)(a2 + 72));
    v58 = sub_ACD640(v57, v27, 0);
    sub_B34240(a2, v65, v14, v58, 0x100u, 0, 0, 0, 0);
    v31 = v65;
  }
  if ( v6 != v69 )
  {
    v32 = *(_BYTE *)(v8 + 104);
    if ( 1LL << v32 )
    {
      _BitScanReverse64(&v33, 1LL << v32);
      v6 &= ~(-1LL << (63 - ((unsigned __int8)v33 ^ 0x3Fu)));
    }
    v34 = *(_QWORD *)(v8 + 136);
    v73 = 257;
    v61 = sub_94B060((unsigned int **)a2, v34, v31, v27, (__int64)v72);
    v35 = sub_AD64C0(*(_QWORD *)(v8 + 136), (unsigned __int8)v6, 0);
    v36 = sub_AA4E30(*(_QWORD *)(a2 + 48));
    v37 = sub_AE5020(v36, *(_QWORD *)(v35 + 8));
    v75 = 257;
    v62 = v37;
    v38 = sub_BD2C40(80, unk_3F10A10);
    v40 = (__int64)v38;
    if ( v38 )
    {
      sub_B4D3C0((__int64)v38, v35, v61, 0, v62, v61, 0, 0);
      v39 = v60;
    }
    (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD, __int64))(**(_QWORD **)(a2 + 88) + 16LL))(
      *(_QWORD *)(a2 + 88),
      v40,
      v74,
      *(_QWORD *)(a2 + 56),
      *(_QWORD *)(a2 + 64),
      v39);
    if ( *(_QWORD *)a2 != *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8) )
    {
      v63 = v8;
      v41 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
      v42 = *(unsigned int **)a2;
      do
      {
        v43 = *((_QWORD *)v42 + 1);
        v44 = *v42;
        v42 += 4;
        sub_B99FD0(v40, v44, v43);
      }
      while ( (unsigned int *)v41 != v42 );
      v8 = v63;
    }
    v45 = *(_QWORD *)(v8 + 128);
    v71 = 257;
    v73 = 257;
    v46 = sub_94BCF0((unsigned int **)a2, a3, v45, (__int64)v70);
    v47 = sub_94B060((unsigned int **)a2, *(_QWORD *)(v8 + 136), (__int64)v46, (int)v69 - 1, (__int64)v72);
    v48 = sub_AA4E30(*(_QWORD *)(a2 + 48));
    v49 = sub_AE5020(v48, *(_QWORD *)(v14 + 8));
    v75 = 257;
    v50 = sub_BD2C40(80, unk_3F10A10);
    v52 = (__int64)v50;
    if ( v50 )
      sub_B4D3C0((__int64)v50, v14, v47, 0, v49, v51, 0, 0);
    (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
      *(_QWORD *)(a2 + 88),
      v52,
      v74,
      *(_QWORD *)(a2 + 56),
      *(_QWORD *)(a2 + 64));
    v53 = *(unsigned int **)a2;
    v54 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
    while ( (unsigned int *)v54 != v53 )
    {
      v55 = *((_QWORD *)v53 + 1);
      v56 = *v53;
      v53 += 4;
      sub_B99FD0(v52, v56, v55);
    }
  }
}
