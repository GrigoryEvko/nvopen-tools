// Function: sub_37830B0
// Address: 0x37830b0
//
void __fastcall sub_37830B0(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v8; // rdx
  unsigned __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // r15
  __int64 v12; // rdx
  unsigned __int16 v13; // cx
  bool v14; // di
  unsigned int v15; // esi
  int v16; // esi
  unsigned int v17; // ebx
  __int16 v18; // ax
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // rcx
  __int64 v22; // rax
  __int64 v23; // rsi
  __int128 v24; // xmm0
  __int128 v25; // xmm1
  __int64 v26; // r8
  __int64 v27; // rbx
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // r8
  __int64 v31; // r9
  __int64 v32; // rax
  unsigned __int8 *v33; // rax
  unsigned int v34; // esi
  int v35; // edx
  __int64 v36; // rdx
  int v37; // edx
  __int64 v38; // rdx
  __int64 v39; // rdx
  unsigned __int64 v40; // rax
  __int64 v41; // [rsp+8h] [rbp-C8h]
  __int64 v42; // [rsp+10h] [rbp-C0h]
  __int64 v43; // [rsp+10h] [rbp-C0h]
  __int64 v44; // [rsp+10h] [rbp-C0h]
  unsigned __int16 v46; // [rsp+20h] [rbp-B0h]
  __int64 *v47; // [rsp+30h] [rbp-A0h]
  unsigned __int8 *v48; // [rsp+40h] [rbp-90h]
  int v49; // [rsp+58h] [rbp-78h]
  unsigned __int16 v50; // [rsp+70h] [rbp-60h] BYREF
  __int64 v51; // [rsp+78h] [rbp-58h]
  unsigned int v52; // [rsp+80h] [rbp-50h] BYREF
  __int64 v53; // [rsp+88h] [rbp-48h]
  __int64 v54; // [rsp+90h] [rbp-40h] BYREF
  int v55; // [rsp+98h] [rbp-38h]

  v8 = *(_QWORD *)(a2 + 48);
  LOWORD(v9) = *(_WORD *)v8;
  v51 = *(_QWORD *)(v8 + 8);
  v10 = *(_QWORD *)(a1 + 8);
  v50 = v9;
  v47 = *(__int64 **)(v10 + 64);
  if ( (_WORD)v9 )
  {
    v11 = 0;
    v12 = (unsigned __int16)v9 - 1;
    v13 = word_4456580[v12];
LABEL_3:
    v14 = (unsigned __int16)(v9 - 176) <= 0x34u;
    v15 = word_4456340[v12];
    LOBYTE(v9) = v14;
    goto LABEL_4;
  }
  v13 = sub_3009970((__int64)&v50, a2, v10, a4, a5);
  LOWORD(v9) = v50;
  v11 = v39;
  if ( v50 )
  {
    v12 = v50 - 1;
    goto LABEL_3;
  }
  v46 = v13;
  v40 = sub_3007240((__int64)&v50);
  v13 = v46;
  v15 = v40;
  v9 = HIDWORD(v40);
  v14 = v9;
LABEL_4:
  v16 = v15 >> 1;
  BYTE4(v54) = v9;
  v17 = v13;
  LODWORD(v54) = v16;
  if ( v14 )
  {
    v18 = sub_2D43AD0(v13, v16);
    v21 = 0;
    if ( v18 )
      goto LABEL_6;
  }
  else
  {
    v18 = sub_2D43050(v13, v16);
    v21 = 0;
    if ( v18 )
      goto LABEL_6;
  }
  v18 = sub_3009450(v47, v17, v11, v54, v19, v20);
  v21 = v38;
LABEL_6:
  LOWORD(v52) = v18;
  v22 = *(_QWORD *)(a2 + 40);
  v23 = *(_QWORD *)(a2 + 80);
  v53 = v21;
  v24 = (__int128)_mm_loadu_si128((const __m128i *)(v22 + 40));
  v25 = (__int128)_mm_loadu_si128((const __m128i *)(v22 + 80));
  v26 = *(_QWORD *)v22;
  v27 = *(_QWORD *)(v22 + 8);
  v54 = v23;
  if ( v23 )
  {
    v42 = v26;
    sub_B96E90((__int64)&v54, v23, 1);
    v26 = v42;
  }
  v41 = v26;
  v55 = *(_DWORD *)(a2 + 72);
  v43 = sub_2E79000(*(__int64 **)(*(_QWORD *)(a1 + 8) + 40LL));
  v32 = sub_3007410((__int64)&v52, *(__int64 **)(*(_QWORD *)(a1 + 8) + 64LL), v28, v29, v30, v31);
  v44 = 1LL << sub_AE5020(v43, v32);
  v33 = sub_3411830(*(__int64 **)(a1 + 8), v52, v53, (__int64)&v54, v41, v27, v24, v25, v44);
  v34 = v52;
  *(_QWORD *)a3 = v33;
  v49 = v35;
  v36 = v53;
  *(_DWORD *)(a3 + 8) = v49;
  v48 = sub_3411830(*(__int64 **)(a1 + 8), v34, v36, (__int64)&v54, (__int64)v33, 1, v24, v25, v44);
  *(_QWORD *)a4 = v48;
  *(_DWORD *)(a4 + 8) = v37;
  sub_3760E70(a1, a2, 1, (unsigned __int64)v48, v27 & 0xFFFFFFFF00000000LL | 1);
  if ( v54 )
    sub_B91220((__int64)&v54, v54);
}
