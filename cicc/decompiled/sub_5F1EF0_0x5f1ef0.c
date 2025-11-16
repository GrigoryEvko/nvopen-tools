// Function: sub_5F1EF0
// Address: 0x5f1ef0
//
__int64 __fastcall sub_5F1EF0(__int64 a1)
{
  __int64 v1; // r12
  __int64 v2; // r13
  __int64 v3; // rbx
  __int64 v4; // rax
  __int64 v5; // rdi
  __int64 v6; // rax
  __int64 v7; // r15
  __int64 v8; // rcx
  __int64 v9; // rdx
  __int64 v10; // rsi
  __int64 v11; // r15
  char v12; // al
  __int64 v13; // rsi
  __int64 v14; // rax
  int v15; // edx
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rsi
  __int64 v19; // rdi
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // r15
  __int64 v23; // rcx
  __int64 v24; // r15
  char v25; // al
  __int64 v26; // rsi
  __int64 v27; // rax
  int v28; // edx
  int v30; // eax
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rcx
  unsigned int v34; // r8d
  __int64 v35; // r15
  char v36; // al
  __int64 v37; // rsi
  __int64 v38; // r14
  int v39; // eax
  _BYTE *v40; // rax
  _BYTE *v41; // rax
  _BYTE *v42; // rax
  __int64 v43; // [rsp-10h] [rbp-A0h]
  __int64 v44; // [rsp-8h] [rbp-98h]
  __int64 v45; // [rsp-8h] [rbp-98h]
  __int64 v46; // [rsp+0h] [rbp-90h]
  __int64 v47; // [rsp+8h] [rbp-88h]
  __int64 v48; // [rsp+8h] [rbp-88h]
  __int64 v49; // [rsp+8h] [rbp-88h]
  __int64 v50; // [rsp+8h] [rbp-88h]
  int v51; // [rsp+8h] [rbp-88h]
  __int64 v52; // [rsp+8h] [rbp-88h]
  __int64 v53; // [rsp+8h] [rbp-88h]
  __int64 v54; // [rsp+8h] [rbp-88h]
  __int64 v55; // [rsp+8h] [rbp-88h]
  unsigned int v56; // [rsp+1Ch] [rbp-74h] BYREF
  __int64 v57; // [rsp+20h] [rbp-70h] BYREF

  v1 = *(_QWORD *)(a1 + 88);
  v2 = *(_QWORD *)(a1 + 96);
  v3 = *(_QWORD *)(*(_QWORD *)(v1 + 168) + 152LL);
  sub_878710(a1, &v57);
  sub_87A680(&v57, &unk_4F077C8, 0);
  v4 = sub_87EBB0(17, v57);
  *(_QWORD *)(v2 + 8) = v4;
  *(_DWORD *)(v4 + 40) = *(_DWORD *)(v3 + 24);
  sub_7296C0(&v56);
  v5 = sub_72CBE0();
  v6 = sub_732700(v5, 0, 0, 0, 0, 0, 0, 0);
  v7 = *(_QWORD *)(v6 + 168);
  v8 = v6;
  *(_BYTE *)(v7 + 17) |= 1u;
  *(_BYTE *)(v7 + 21) |= 1u;
  *(_QWORD *)(v7 + 40) = v1;
  v9 = v44;
  v10 = dword_4D048B8;
  if ( dword_4D048B8 )
  {
    v54 = v6;
    v41 = (_BYTE *)sub_725E60(v5, dword_4D048B8, v44);
    v8 = v54;
    *v41 |= 9u;
    *(_QWORD *)(v7 + 56) = v41;
  }
  v47 = v8;
  v11 = sub_725FD0(v5, v10, v9);
  *(_QWORD *)(v11 + 152) = v47;
  sub_725ED0(v11, 1);
  *(_DWORD *)(v11 + 192) |= 0x21000u;
  v12 = *(_BYTE *)(v11 + 195);
  if ( *(char *)(v1 + 177) < 0 )
  {
    v12 |= 8u;
    *(_BYTE *)(v11 + 195) = v12;
  }
  *(_BYTE *)(v11 + 195) = v12 | 0x10;
  v13 = v57;
  *(_QWORD *)(v11 + 112) = *(_QWORD *)(v3 + 144);
  *(_QWORD *)(v3 + 144) = v11;
  v14 = sub_87EBB0(10, v13);
  v15 = *(_DWORD *)(v3 + 24);
  *(_QWORD *)(v14 + 88) = v11;
  *(_DWORD *)(v14 + 40) = v15;
  *(_QWORD *)v11 = v14;
  v48 = v14;
  sub_877D50(v11, *(_QWORD *)v14);
  sub_877E20(v48, v11, v1);
  *(_BYTE *)(v11 + 89) = *(_BYTE *)(v1 + 89) & 1 | *(_BYTE *)(v11 + 89) & 0xFE;
  v16 = *(_QWORD *)(v2 + 8);
  *(_QWORD *)(v2 + 16) = v48;
  *(_QWORD *)(v16 + 88) = v48;
  sub_736C90(v11, 1);
  if ( unk_4F068EC )
    sub_89A080(v11);
  v17 = sub_73C570(v1, 1, -1);
  v18 = sub_72D600(v17);
  v19 = sub_72CBE0();
  v20 = sub_732700(v19, v18, 0, 0, 0, 0, 0, 0);
  v22 = *(_QWORD *)(v20 + 168);
  v23 = v20;
  *(_BYTE *)(v22 + 17) |= 1u;
  *(_BYTE *)(v22 + 21) |= 1u;
  *(_QWORD *)(v22 + 40) = v1;
  if ( dword_4D048B8 )
  {
    v53 = v20;
    v40 = (_BYTE *)sub_725E60(v19, v18, v21);
    v23 = v53;
    *v40 |= 9u;
    *(_QWORD *)(v22 + 56) = v40;
  }
  v49 = v23;
  v24 = sub_725FD0(v19, v18, v21);
  *(_QWORD *)(v24 + 152) = v49;
  sub_725ED0(v24, 1);
  *(_DWORD *)(v24 + 192) = *(_DWORD *)(v24 + 192) & 0xFFFBEDFF | (word_4D04898 << 9) & 0x200 | 0x41000;
  v25 = *(_BYTE *)(v24 + 195);
  if ( *(char *)(v1 + 177) < 0 )
  {
    v25 |= 8u;
    *(_BYTE *)(v24 + 195) = v25;
  }
  *(_BYTE *)(v24 + 195) = v25 | 0x10;
  v26 = v57;
  *(_QWORD *)(v24 + 112) = *(_QWORD *)(v3 + 144);
  *(_QWORD *)(v3 + 144) = v24;
  v27 = sub_87EBB0(10, v26);
  v28 = *(_DWORD *)(v3 + 24);
  *(_QWORD *)(v27 + 88) = v24;
  *(_DWORD *)(v27 + 40) = v28;
  *(_QWORD *)v24 = v27;
  v50 = v27;
  sub_877D50(v24, *(_QWORD *)v27);
  sub_877E20(v50, v24, v1);
  *(_BYTE *)(v24 + 89) = *(_BYTE *)(v1 + 89) & 1 | *(_BYTE *)(v24 + 89) & 0xFE;
  *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v2 + 8) + 88LL) + 8LL) = v50;
  sub_736C90(v24, 1);
  if ( unk_4F068EC )
    sub_89A080(v24);
  if ( dword_4D0446C )
  {
    v51 = sub_72D6A0(v1);
    v30 = sub_72CBE0();
    v31 = sub_732700(v30, v51, 0, 0, 0, 0, 0, 0);
    v32 = *(_QWORD *)(v31 + 168);
    v33 = v31;
    *(_BYTE *)(v32 + 17) |= 1u;
    *(_BYTE *)(v32 + 21) |= 1u;
    v34 = dword_4D048B8;
    *(_QWORD *)(v32 + 40) = v1;
    if ( v34 )
    {
      v46 = v32;
      v55 = v31;
      v42 = (_BYTE *)sub_725E60(v45, v43, v32);
      v32 = v46;
      v33 = v55;
      *v42 |= 9u;
      *(_QWORD *)(v46 + 56) = v42;
    }
    v52 = v33;
    v35 = sub_725FD0(v45, v43, v32);
    *(_QWORD *)(v35 + 152) = v52;
    sub_725ED0(v35, 1);
    *(_DWORD *)(v35 + 192) = *(_DWORD *)(v35 + 192) & 0xFFFBEDFF | (word_4D04898 << 9) & 0x200 | 0x41000;
    v36 = *(_BYTE *)(v35 + 195);
    if ( *(char *)(v1 + 177) < 0 )
    {
      v36 |= 8u;
      *(_BYTE *)(v35 + 195) = v36;
    }
    *(_BYTE *)(v35 + 195) = v36 | 0x10;
    v37 = v57;
    *(_QWORD *)(v35 + 112) = *(_QWORD *)(v3 + 144);
    *(_QWORD *)(v3 + 144) = v35;
    v38 = sub_87EBB0(10, v37);
    v39 = *(_DWORD *)(v3 + 24);
    *(_QWORD *)(v38 + 88) = v35;
    *(_DWORD *)(v38 + 40) = v39;
    *(_QWORD *)v35 = v38;
    sub_877D50(v35, *(_QWORD *)v38);
    sub_877E20(v38, v35, v1);
    *(_BYTE *)(v35 + 89) = *(_BYTE *)(v1 + 89) & 1 | *(_BYTE *)(v35 + 89) & 0xFE;
    *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v2 + 8) + 88LL) + 8LL) + 8LL) = v38;
    sub_736C90(v35, 1);
    if ( unk_4F068EC )
      sub_89A080(v35);
  }
  sub_877E20(*(_QWORD *)(v2 + 8), 0, v1);
  sub_886160(*(_QWORD *)(v2 + 8));
  sub_729730(v56);
  return *(_QWORD *)(v2 + 8);
}
