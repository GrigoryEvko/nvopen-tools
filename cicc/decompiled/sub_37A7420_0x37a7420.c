// Function: sub_37A7420
// Address: 0x37a7420
//
__m128i *__fastcall sub_37A7420(_QWORD *a1, __int64 a2, __m128i a3)
{
  __int16 *v4; // rax
  __int16 v5; // dx
  __int64 v6; // rax
  __int64 v7; // rax
  unsigned __int64 v8; // rsi
  unsigned __int64 v9; // rdx
  unsigned __int64 v10; // r13
  unsigned __int64 v11; // r12
  __int64 v12; // rax
  unsigned __int16 v13; // dx
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rbx
  __int64 v18; // rdx
  __int64 v19; // rax
  unsigned __int64 v20; // rbx
  int v21; // ecx
  __int64 v22; // rdx
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r8
  __m128i *v26; // r12
  __int64 v28; // rdx
  unsigned __int16 v29; // ax
  __int64 v30; // r15
  __int64 v31; // rdx
  unsigned __int16 v32; // bx
  __int64 v33; // rdx
  __int64 v34; // rax
  __int64 v35; // rdx
  int v36; // ebx
  unsigned __int64 v37; // rax
  unsigned __int64 v38; // rcx
  unsigned int v39; // eax
  __int64 v40; // rdx
  int v41; // r9d
  unsigned __int8 *v42; // rax
  _QWORD *v43; // r14
  __int64 v44; // rdx
  __int64 v45; // r13
  unsigned __int8 *v46; // r12
  __int128 v47; // rax
  __int64 v48; // r9
  unsigned __int8 *v49; // rax
  unsigned __int64 v50; // rax
  unsigned int v51; // eax
  __int64 v52; // rdx
  int v53; // r9d
  unsigned __int8 *v54; // rax
  _QWORD *v55; // r14
  __int64 v56; // rdx
  __int64 v57; // r13
  unsigned __int8 *v58; // r12
  __int128 v59; // rax
  __int64 v60; // r9
  __int64 v61; // rdx
  __int64 v62; // rcx
  __int64 v63; // r8
  __int128 v64; // [rsp-30h] [rbp-110h]
  __int128 v65; // [rsp-30h] [rbp-110h]
  char v66; // [rsp+8h] [rbp-D8h]
  __int64 v67; // [rsp+18h] [rbp-C8h]
  unsigned int v68; // [rsp+20h] [rbp-C0h] BYREF
  unsigned __int64 v69; // [rsp+28h] [rbp-B8h]
  unsigned __int16 v70; // [rsp+30h] [rbp-B0h] BYREF
  __int64 v71; // [rsp+38h] [rbp-A8h]
  unsigned __int64 v72; // [rsp+40h] [rbp-A0h] BYREF
  int v73; // [rsp+48h] [rbp-98h]
  unsigned int v74; // [rsp+50h] [rbp-90h] BYREF
  __int64 v75; // [rsp+58h] [rbp-88h]
  unsigned __int64 v76; // [rsp+60h] [rbp-80h]
  __int64 v77; // [rsp+68h] [rbp-78h]
  __int64 v78; // [rsp+70h] [rbp-70h]
  __int64 v79; // [rsp+78h] [rbp-68h]
  __int64 v80; // [rsp+80h] [rbp-60h]
  __int64 v81; // [rsp+88h] [rbp-58h]
  _QWORD v82[10]; // [rsp+90h] [rbp-50h] BYREF

  v4 = *(__int16 **)(a2 + 48);
  v5 = *v4;
  v69 = *((_QWORD *)v4 + 1);
  v6 = *(_QWORD *)(a2 + 40);
  LOWORD(v68) = v5;
  v7 = sub_379AB60((__int64)a1, *(_QWORD *)v6, *(_QWORD *)(v6 + 8));
  v8 = *(_QWORD *)(a2 + 80);
  v10 = v9;
  v11 = v7;
  v12 = *(_QWORD *)(v7 + 48) + 16LL * (unsigned int)v9;
  v13 = *(_WORD *)v12;
  v14 = *(_QWORD *)(v12 + 8);
  v72 = v8;
  v70 = v13;
  v71 = v14;
  if ( v8 )
  {
    sub_B96E90((__int64)&v72, v8, 1);
    v13 = v70;
  }
  v73 = *(_DWORD *)(a2 + 72);
  if ( v13 )
  {
    if ( v13 == 1 || (unsigned __int16)(v13 - 504) <= 7u )
      goto LABEL_55;
    v20 = *(_QWORD *)&byte_444C4A0[16 * v13 - 16];
    v66 = byte_444C4A0[16 * v13 - 8];
  }
  else
  {
    v15 = sub_3007260((__int64)&v70);
    v17 = v16;
    v18 = v15;
    v19 = v17;
    v78 = v18;
    v20 = v18;
    v79 = v19;
    v66 = v19;
  }
  v21 = (unsigned __int16)v68;
  if ( (_WORD)v68 )
  {
    if ( (_WORD)v68 == 1 || (unsigned __int16)(v68 - 504) <= 7u )
      goto LABEL_55;
    if ( (unsigned __int16)(v68 - 17) <= 0xD3u )
      goto LABEL_20;
    if ( (_WORD)v68 == 261 )
      goto LABEL_9;
    v8 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v68 - 16];
    if ( byte_444C4A0[16 * (unsigned __int16)v68 - 8] != v66 || v20 % v8 )
    {
LABEL_19:
      if ( (unsigned __int16)(v21 - 17) > 0xD3u )
        goto LABEL_9;
LABEL_20:
      v28 = 0;
      v29 = word_4456580[v21 - 1];
      goto LABEL_21;
    }
LABEL_43:
    v50 = v20 / v8;
    v8 = v68;
    v51 = sub_327FCF0(*(__int64 **)(a1[1] + 64LL), v68, v69, v50, 0);
    if ( (_WORD)v51 )
    {
      v8 = (unsigned __int16)v51;
      if ( *(_QWORD *)(*a1 + 8LL * (unsigned __int16)v51 + 112) )
      {
        v54 = sub_33FAF80(a1[1], 234, (__int64)&v72, v51, v52, v53, a3);
        v55 = (_QWORD *)a1[1];
        v57 = v56;
        v58 = v54;
        *(_QWORD *)&v59 = sub_3400EE0((__int64)v55, 0, (__int64)&v72, 0, a3);
        *((_QWORD *)&v65 + 1) = v57;
        *(_QWORD *)&v65 = v58;
        v49 = sub_3406EB0(v55, 0x9Eu, (__int64)&v72, v68, v69, v60, v65, v59);
        goto LABEL_34;
      }
    }
    v21 = (unsigned __int16)v68;
    if ( !(_WORD)v68 )
    {
LABEL_8:
      if ( !sub_30070B0((__int64)&v68) )
      {
LABEL_9:
        v26 = sub_375AC00((__int64)a1, v11, v10, v68, v69);
        goto LABEL_10;
      }
      goto LABEL_38;
    }
    goto LABEL_19;
  }
  v76 = sub_3007260((__int64)&v68);
  v77 = v22;
  if ( !sub_30070B0((__int64)&v68) )
  {
    v8 = v76;
    if ( v66 != (_BYTE)v77 || v20 % v76 )
      goto LABEL_8;
    goto LABEL_43;
  }
LABEL_38:
  v29 = sub_3009970((__int64)&v68, v8, v23, v24, v25);
LABEL_21:
  LOWORD(v74) = v29;
  v75 = v28;
  if ( v29 )
  {
    if ( v29 == 1 || (unsigned __int16)(v29 - 504) <= 7u )
      goto LABEL_55;
    v30 = *(_QWORD *)&byte_444C4A0[16 * v29 - 16];
  }
  else
  {
    v80 = sub_3007260((__int64)&v74);
    LODWORD(v30) = v80;
    v81 = v31;
  }
  if ( v20 % (unsigned int)v30 )
    goto LABEL_9;
  v32 = v70;
  if ( v70 )
  {
    if ( (unsigned __int16)(v70 - 17) > 0xD3u )
    {
LABEL_26:
      v33 = v71;
      goto LABEL_27;
    }
    v32 = word_4456580[v70 - 1];
    v33 = 0;
  }
  else
  {
    if ( !sub_30070B0((__int64)&v70) )
      goto LABEL_26;
    v32 = sub_3009970((__int64)&v70, v8, v61, v62, v63);
  }
LABEL_27:
  LOWORD(v82[0]) = v32;
  v82[1] = v33;
  if ( !v32 )
  {
    v34 = sub_3007260((__int64)v82);
    v82[2] = v34;
    v82[3] = v35;
    goto LABEL_29;
  }
  if ( v32 == 1 || (unsigned __int16)(v32 - 504) <= 7u )
LABEL_55:
    BUG();
  v34 = *(_QWORD *)&byte_444C4A0[16 * v32 - 16];
LABEL_29:
  v36 = v34;
  if ( v70 )
  {
    LODWORD(v37) = word_4456340[v70 - 1];
    LOBYTE(v38) = (unsigned __int16)(v70 - 176) <= 0x34u;
  }
  else
  {
    v37 = sub_3007240((__int64)&v70);
    v82[0] = v37;
    v38 = HIDWORD(v37);
  }
  BYTE4(v67) = v38;
  LODWORD(v67) = v36 * (int)v37 / (unsigned int)v30;
  v39 = sub_327FD70(*(__int64 **)(a1[1] + 64LL), v74, v75, v67);
  if ( !(_WORD)v39 || !*(_QWORD *)(*a1 + 8LL * (unsigned __int16)v39 + 112) )
    goto LABEL_9;
  v42 = sub_33FAF80(a1[1], 234, (__int64)&v72, v39, v40, v41, a3);
  v43 = (_QWORD *)a1[1];
  v45 = v44;
  v46 = v42;
  *(_QWORD *)&v47 = sub_3400EE0((__int64)v43, 0, (__int64)&v72, 0, a3);
  *((_QWORD *)&v64 + 1) = v45;
  *(_QWORD *)&v64 = v46;
  v49 = sub_3406EB0(v43, 0xA1u, (__int64)&v72, v68, v69, v48, v64, v47);
LABEL_34:
  v26 = (__m128i *)v49;
LABEL_10:
  if ( v72 )
    sub_B91220((__int64)&v72, v72);
  return v26;
}
