// Function: sub_2E81690
// Address: 0x2e81690
//
void __fastcall sub_2E81690(__int64 a1)
{
  __int64 *v1; // r15
  __int64 v3; // rax
  __int64 v4; // r12
  __int64 v5; // rdi
  __int64 (*v6)(); // rax
  int v7; // r14d
  char v8; // r13
  unsigned __int16 v9; // ax
  char v10; // r12
  __int64 (*v11)(); // rax
  __int64 v12; // rax
  __int64 v13; // r12
  __int64 v14; // rax
  __int64 v15; // rdi
  __int64 (*v16)(); // rax
  __int64 v17; // rax
  __int64 v18; // rdi
  __int64 v19; // rsi
  __int64 v20; // rdi
  char v21; // dl
  unsigned __int64 v22; // rax
  __int64 v23; // r8
  __int64 v24; // rax
  __int64 v25; // r12
  __int64 v26; // rdi
  int v27; // eax
  __int64 v28; // r13
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // r8
  __int64 v33; // r9
  __int64 v34; // r12
  unsigned __int64 v35; // rdi
  __int64 (*v36)(); // rax
  unsigned __int8 v37; // al
  _OWORD *v38; // rax
  __int64 v39; // r12
  unsigned __int8 v40; // al
  __int64 v41; // [rsp+8h] [rbp-38h]

  v1 = (__int64 *)(a1 + 128);
  *(_QWORD *)(a1 + 344) |= 5uLL;
  v3 = sub_A777F0(0x200u, (__int64 *)(a1 + 128));
  v4 = v3;
  if ( v3 )
    sub_2EBE060(v3, a1);
  v5 = *(_QWORD *)(a1 + 16);
  *(_QWORD *)(a1 + 32) = v4;
  *(_QWORD *)(a1 + 40) = 0;
  v6 = *(__int64 (**)())(*(_QWORD *)v5 + 136LL);
  if ( v6 == sub_2DD19D0 )
    BUG();
  LOBYTE(v7) = *(_BYTE *)(v6() + 20);
  if ( (_BYTE)v7 )
    v7 = sub_B2D620(*(_QWORD *)a1, "no-realign-stack", 0x10u) ^ 1;
  v8 = sub_B2D610(*(_QWORD *)a1, 94);
  if ( !v8 )
    v8 = sub_B2D620(*(_QWORD *)a1, "stackrealign", 0xCu);
  v41 = *(_QWORD *)(a1 + 16);
  v9 = sub_A74690((_QWORD *)(*(_QWORD *)a1 + 120LL));
  v10 = v9;
  if ( !HIBYTE(v9) )
  {
    v11 = *(__int64 (**)())(*(_QWORD *)v41 + 136LL);
    if ( v11 == sub_2DD19D0 )
      BUG();
    v10 = *(_BYTE *)(((__int64 (__fastcall *)(__int64))v11)(v41) + 12);
  }
  v12 = sub_A777F0(0x2B8u, v1);
  if ( v12 )
  {
    *(_BYTE *)v12 = v10;
    *(_BYTE *)(v12 + 1) = v7;
    *(_BYTE *)(v12 + 2) = v7 & v8;
    *(_QWORD *)(v12 + 8) = 0;
    *(_QWORD *)(v12 + 16) = 0;
    *(_QWORD *)(v12 + 24) = 0;
    *(_QWORD *)(v12 + 32) = 0;
    *(_BYTE *)(v12 + 40) = 0;
    *(_QWORD *)(v12 + 48) = 0;
    *(_QWORD *)(v12 + 56) = 0;
    *(_WORD *)(v12 + 64) = 0;
    *(_QWORD *)(v12 + 128) = v12 + 144;
    *(_BYTE *)(v12 + 66) = 0;
    *(_QWORD *)(v12 + 68) = -1;
    *(_QWORD *)(v12 + 80) = -1;
    *(_DWORD *)(v12 + 88) = 0;
    *(_QWORD *)(v12 + 96) = 0;
    *(_QWORD *)(v12 + 104) = 0;
    *(_QWORD *)(v12 + 112) = 0;
    *(_BYTE *)(v12 + 120) = 0;
    *(_QWORD *)(v12 + 136) = 0x2000000000LL;
    *(_QWORD *)(v12 + 656) = 0;
    *(_DWORD *)(v12 + 664) = 0;
    *(_WORD *)(v12 + 668) = 0;
    *(_BYTE *)(v12 + 670) = 0;
    *(_QWORD *)(v12 + 672) = 0;
    *(_QWORD *)(v12 + 680) = 0;
    *(_QWORD *)(v12 + 688) = 0;
  }
  *(_QWORD *)(a1 + 48) = v12;
  sub_2E78940(*(_QWORD *)a1, v12);
  if ( (unsigned __int8)sub_B2D610(*(_QWORD *)a1, 94) )
  {
    v39 = *(_QWORD *)(a1 + 48);
    v40 = sub_A74690((_QWORD *)(*(_QWORD *)a1 + 120LL));
    sub_2E76F70(v39, v40);
  }
  v13 = sub_2E79000((__int64 *)a1);
  v14 = sub_A777F0(0x48u, v1);
  if ( v14 )
  {
    *(_BYTE *)v14 = 0;
    *(_QWORD *)(v14 + 8) = 0;
    *(_QWORD *)(v14 + 16) = 0;
    *(_QWORD *)(v14 + 24) = 0;
    *(_QWORD *)(v14 + 32) = 0;
    *(_QWORD *)(v14 + 40) = 0;
    *(_QWORD *)(v14 + 48) = 0;
    *(_DWORD *)(v14 + 56) = 0;
    *(_QWORD *)(v14 + 64) = v13;
  }
  v15 = *(_QWORD *)(a1 + 16);
  *(_QWORD *)(a1 + 56) = v14;
  v16 = *(__int64 (**)())(*(_QWORD *)v15 + 144LL);
  if ( v16 == sub_2C8F680 )
    BUG();
  v17 = v16();
  v18 = *(_QWORD *)a1;
  v19 = 47;
  *(_BYTE *)(a1 + 340) = *(_BYTE *)(v17 + 74);
  if ( !(unsigned __int8)sub_B2D610(v18, 47) )
  {
    v36 = *(__int64 (**)())(**(_QWORD **)(a1 + 16) + 144LL);
    if ( v36 == sub_2C8F680 )
      BUG();
    v37 = *(_BYTE *)(v36() + 75);
    if ( *(_BYTE *)(a1 + 340) < v37 )
      *(_BYTE *)(a1 + 340) = v37;
  }
  v20 = *(_QWORD *)a1;
  if ( (*(_BYTE *)(*(_QWORD *)a1 + 7LL) & 0x20) != 0 )
  {
    v19 = 32;
    if ( sub_B91C10(v20, 32) )
      goto LABEL_21;
    v20 = *(_QWORD *)a1;
    if ( (*(_BYTE *)(*(_QWORD *)a1 + 7LL) & 0x20) != 0 )
    {
      v19 = 36;
      if ( !sub_B91C10(v20, 36) )
      {
LABEL_23:
        v20 = *(_QWORD *)a1;
        goto LABEL_24;
      }
LABEL_21:
      if ( *(_BYTE *)(a1 + 340) <= 1u )
        *(_BYTE *)(a1 + 340) = 2;
      goto LABEL_23;
    }
  }
LABEL_24:
  if ( (_DWORD)qword_501FFC8 )
  {
    v21 = -1;
    if ( 1LL << qword_501FFC8 )
    {
      _BitScanReverse64(&v22, 1LL << qword_501FFC8);
      v21 = 63 - (v22 ^ 0x3F);
    }
    *(_BYTE *)(a1 + 340) = v21;
  }
  *(_QWORD *)(a1 + 64) = 0;
  v23 = 0;
  if ( (*(_BYTE *)(v20 + 2) & 8) != 0 )
    v23 = sub_B2E500(v20);
  if ( (unsigned int)sub_B2A630(v23) - 7 <= 3 )
  {
    v19 = (__int64)v1;
    v24 = sub_A777F0(0x2F8u, v1);
    v25 = v24;
    if ( v24 )
      sub_3012440(v24);
    *(_QWORD *)(a1 + 88) = v25;
  }
  v26 = 0;
  if ( (*(_BYTE *)(*(_QWORD *)a1 + 2LL) & 8) != 0 )
    v26 = sub_B2E500(*(_QWORD *)a1);
  v27 = sub_B2A630(v26);
  if ( v27 > 10 )
  {
    if ( v27 != 12 )
      goto LABEL_38;
  }
  else if ( v27 <= 6 )
  {
    goto LABEL_38;
  }
  v19 = (__int64)v1;
  v38 = (_OWORD *)sub_A777F0(0x40u, v1);
  if ( v38 )
  {
    *v38 = 0;
    v38[1] = 0;
    v38[2] = 0;
    v38[3] = 0;
  }
  *(_QWORD *)(a1 + 80) = v38;
LABEL_38:
  v28 = *(_QWORD *)(a1 + 8);
  v29 = sub_22077B0(0xF0u);
  v34 = v29;
  if ( v29 )
  {
    v19 = v28;
    sub_2F3F3E0(v29, v28);
  }
  v35 = *(_QWORD *)(a1 + 352);
  *(_QWORD *)(a1 + 352) = v34;
  if ( v35 )
    sub_2E81360(v35, v19, v30, v31, v32, v33);
}
