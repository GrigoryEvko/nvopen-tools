// Function: sub_32899A0
// Address: 0x32899a0
//
__int64 __fastcall sub_32899A0(__int64 *a1, __int64 a2)
{
  __int64 v2; // r15
  __int16 *v4; // rax
  __int16 v5; // dx
  __int64 v6; // rax
  void *v7; // rax
  __int64 result; // rax
  __int64 v9; // rdi
  __int64 v10; // r12
  bool (__fastcall *v11)(__int64, __int64); // rax
  __int64 v12; // rsi
  int v13; // eax
  unsigned __int16 v14; // r12
  __int64 v15; // rdx
  __int64 v16; // rsi
  __int64 v17; // rdx
  __int64 v18; // r8
  __int64 v19; // rax
  __int16 v20; // cx
  __int64 v21; // rdx
  __int64 v22; // r13
  __int64 v23; // r12
  unsigned __int8 v24; // al
  __int64 v25; // rdx
  __int64 v26; // r15
  __int64 v27; // r14
  __int128 v28; // rax
  int v29; // r9d
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // r15
  __int64 v33; // r14
  __int128 v34; // rax
  int v35; // r9d
  __int64 v36; // rax
  __int64 v37; // rdx
  __int64 v38; // rdx
  __int64 v39; // rcx
  __int64 v40; // r8
  bool v41; // al
  unsigned __int64 v42; // rax
  unsigned __int64 v43; // rsi
  unsigned __int64 v44; // rax
  bool v45; // dl
  __int64 *v46; // r12
  __int64 v47; // r8
  __int64 v48; // r9
  __int64 v49; // rax
  __int64 v50; // rdx
  __int128 v51; // [rsp-20h] [rbp-120h]
  __int128 v52; // [rsp-10h] [rbp-110h]
  __int64 v53; // [rsp+0h] [rbp-100h]
  __int64 v54; // [rsp+8h] [rbp-F8h]
  __int16 v55; // [rsp+8h] [rbp-F8h]
  __int64 v56; // [rsp+10h] [rbp-F0h]
  __int64 v57; // [rsp+10h] [rbp-F0h]
  __int64 v58; // [rsp+18h] [rbp-E8h] BYREF
  _QWORD v59[2]; // [rsp+20h] [rbp-E0h] BYREF
  unsigned int v60; // [rsp+30h] [rbp-D0h] BYREF
  __int64 v61; // [rsp+38h] [rbp-C8h]
  __int64 v62; // [rsp+40h] [rbp-C0h] BYREF
  __int64 v63; // [rsp+48h] [rbp-B8h]
  __int64 v64; // [rsp+50h] [rbp-B0h] BYREF
  __int64 v65; // [rsp+58h] [rbp-A8h]
  __int64 v66; // [rsp+60h] [rbp-A0h] BYREF
  int v67; // [rsp+68h] [rbp-98h]
  __int64 v68; // [rsp+70h] [rbp-90h]
  __int64 v69; // [rsp+78h] [rbp-88h]
  __int128 v70; // [rsp+80h] [rbp-80h] BYREF
  __int64 *v71[12]; // [rsp+A0h] [rbp-60h] BYREF

  v4 = *(__int16 **)(a2 + 48);
  v58 = a2;
  v5 = *v4;
  v6 = *((_QWORD *)v4 + 1);
  LOWORD(v60) = v5;
  v61 = v6;
  v7 = sub_300AC80((unsigned __int16 *)&v60, a2);
  if ( !sub_C33700(v7) )
    return 0;
  v62 = 0;
  v71[0] = &v58;
  v71[1] = &v62;
  v71[2] = &v64;
  LODWORD(v63) = 0;
  v64 = 0;
  LODWORD(v65) = 0;
  v59[0] = 0;
  v71[3] = a1;
  v71[4] = v59;
  if ( !(unsigned __int8)sub_3269C40(v71, 0) && !(unsigned __int8)sub_3269C40(v71, 1u) )
    return 0;
  v9 = a1[1];
  v10 = v58;
  v11 = *(bool (__fastcall **)(__int64, __int64))(*(_QWORD *)v9 + 440LL);
  if ( v11 == sub_2FE3070 )
  {
    if ( *(_DWORD *)(v58 + 24) == 99 )
      goto LABEL_7;
    return 0;
  }
  if ( !((unsigned __int8 (__fastcall *)(__int64, __int64, __int64, __int64, __int64, __int64))v11)(
          v9,
          v58,
          v62,
          v63,
          v64,
          v65) )
    return 0;
  v10 = v58;
LABEL_7:
  v12 = *(_QWORD *)(v10 + 80);
  v66 = v12;
  if ( v12 )
    sub_B96E90((__int64)&v66, v12, 1);
  v13 = *(_DWORD *)(v10 + 72);
  v14 = v60;
  v67 = v13;
  if ( (_WORD)v60 )
  {
    if ( (unsigned __int16)(v60 - 17) > 0xD3u )
    {
LABEL_11:
      v15 = v61;
      goto LABEL_12;
    }
    v14 = word_4456580[(unsigned __int16)v60 - 1];
    v15 = 0;
  }
  else
  {
    if ( !sub_30070B0((__int64)&v60) )
      goto LABEL_11;
    v14 = sub_3009970((__int64)&v60, v12, v38, v39, v40);
  }
LABEL_12:
  LOWORD(v70) = v14;
  *((_QWORD *)&v70 + 1) = v15;
  if ( v14 )
  {
    if ( v14 == 1 || (unsigned __int16)(v14 - 504) <= 7u )
      BUG();
    v16 = *(_QWORD *)&byte_444C4A0[16 * v14 - 16];
  }
  else
  {
    v68 = sub_3007260((__int64)&v70);
    LODWORD(v16) = v68;
    v69 = v17;
  }
  v18 = *a1;
  switch ( (_DWORD)v16 )
  {
    case 1:
      v20 = 2;
      break;
    case 2:
      v20 = 3;
      break;
    case 4:
      v20 = 4;
      break;
    case 8:
      v20 = 5;
      break;
    case 0x10:
      v20 = 6;
      break;
    case 0x20:
      v20 = 7;
      break;
    case 0x40:
      v20 = 8;
      break;
    case 0x80:
      v20 = 9;
      break;
    default:
      v19 = sub_3007020(*(_QWORD **)(v18 + 64), v16);
      v18 = *a1;
      v2 = v19;
      v20 = v19;
      v22 = v21;
      goto LABEL_23;
  }
  v22 = 0;
LABEL_23:
  LOWORD(v2) = v20;
  v23 = v2;
  if ( (_WORD)v60 )
  {
    if ( (unsigned __int16)(v60 - 17) > 0xD3u )
      goto LABEL_25;
    v45 = (unsigned __int16)(v60 - 176) <= 0x34u;
    LODWORD(v43) = word_4456340[(unsigned __int16)v60 - 1];
    LOBYTE(v44) = v45;
  }
  else
  {
    v55 = v20;
    v57 = v18;
    v41 = sub_30070B0((__int64)&v60);
    v18 = v57;
    v20 = v55;
    if ( !v41 )
      goto LABEL_25;
    v42 = sub_3007240((__int64)&v60);
    v18 = v57;
    v43 = v42;
    v44 = HIDWORD(v42);
    v59[1] = v43;
    v45 = v44;
  }
  v46 = *(__int64 **)(v18 + 64);
  LODWORD(v70) = v43;
  BYTE4(v70) = v44;
  if ( v45 )
    v20 = sub_2D43AD0(v2, v43);
  else
    v20 = sub_2D43050(v2, v43);
  if ( v20 )
  {
    v22 = 0;
  }
  else
  {
    v53 = sub_3009450(v46, (unsigned int)v2, v22, v70, v47, v48);
    v20 = v53;
    v22 = v50;
  }
  v49 = v53;
  v18 = *a1;
  LOWORD(v49) = v20;
  v23 = v49;
LABEL_25:
  LOWORD(v23) = v20;
  *(_QWORD *)&v70 = v23;
  *((_QWORD *)&v70 + 1) = v22;
  v24 = sub_33DE9F0(v18, v64, v65, 0);
  result = sub_3289780(a1, v64, v65, (__int64)&v66, v24, 1, v70, 1);
  v26 = v25;
  v27 = result;
  if ( result )
  {
    *(_QWORD *)&v28 = sub_3400E40(*a1, SLODWORD(v59[0]), (unsigned int)v23, v22, &v66);
    *((_QWORD *)&v51 + 1) = v26;
    *(_QWORD *)&v51 = v27;
    v30 = sub_3406EB0(*a1, 190, (unsigned int)&v66, v23, v22, v29, v51, v28);
    v32 = v31;
    v54 = *a1;
    v33 = v30;
    *(_QWORD *)&v34 = sub_33FB890(*a1, (unsigned int)v23, v22, v62, v63);
    *((_QWORD *)&v52 + 1) = v32;
    *(_QWORD *)&v52 = v33;
    v36 = sub_3406EB0(
            v54,
            (unsigned int)(*(_DWORD *)(v58 + 24) != 98) + 56,
            (unsigned int)&v66,
            v23,
            v22,
            v35,
            v34,
            v52);
    result = sub_33FB890(*a1, v60, v61, v36, v37);
  }
  if ( v66 )
  {
    v56 = result;
    sub_B91220((__int64)&v66, v66);
    return v56;
  }
  return result;
}
