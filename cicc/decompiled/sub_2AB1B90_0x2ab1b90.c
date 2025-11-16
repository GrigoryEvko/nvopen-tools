// Function: sub_2AB1B90
// Address: 0x2ab1b90
//
__int64 __fastcall sub_2AB1B90(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rdx
  _BYTE *v8; // rdi
  _BYTE *v9; // rsi
  __int64 v10; // r10
  __int64 v11; // rcx
  __int64 v12; // rdx
  unsigned __int64 v13; // r13
  __int64 v14; // rax
  __int64 v15; // rcx
  __int64 v16; // rsi
  __int64 v17; // rsi
  __int64 v18; // rax
  __int64 v19; // rax
  unsigned __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rax
  _BYTE *v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  _BYTE *v26; // rax
  __int64 v27; // r8
  __int64 v28; // r9
  __int64 v29; // rax
  _BYTE *v30; // rsi
  __int64 v31; // rdi
  __int64 v32; // rdi
  __int64 v33; // rax
  __int64 v34; // rax
  _BYTE *v35; // rax
  __int64 v36; // rax
  __int64 v37; // rdx
  __int64 v38; // rdx
  _BYTE *v39; // rdx
  __int64 v40; // rdx
  __int64 v41; // rax
  __int64 v42; // rax
  _BYTE *v43; // rax
  __int64 v44; // rax
  __int64 v45; // rax
  _BYTE *v46; // rax
  __int64 v47; // rax
  __int64 v48; // rax
  _BYTE *v49; // rax
  __int64 v51; // [rsp+10h] [rbp-360h]
  __int64 v52; // [rsp+20h] [rbp-350h]
  _BYTE *v53; // [rsp+20h] [rbp-350h]
  _BYTE v55[32]; // [rsp+40h] [rbp-330h] BYREF
  _BYTE v56[64]; // [rsp+60h] [rbp-310h] BYREF
  __int64 v57; // [rsp+A0h] [rbp-2D0h]
  __int64 v58; // [rsp+A8h] [rbp-2C8h]
  _BYTE *v59; // [rsp+B0h] [rbp-2C0h]
  _BYTE v60[32]; // [rsp+C0h] [rbp-2B0h] BYREF
  _BYTE v61[64]; // [rsp+E0h] [rbp-290h] BYREF
  __int64 v62; // [rsp+120h] [rbp-250h]
  __int64 v63; // [rsp+128h] [rbp-248h]
  unsigned __int64 v64; // [rsp+130h] [rbp-240h]
  _BYTE v65[32]; // [rsp+140h] [rbp-230h] BYREF
  _BYTE v66[64]; // [rsp+160h] [rbp-210h] BYREF
  __int64 v67; // [rsp+1A0h] [rbp-1D0h]
  __int64 v68; // [rsp+1A8h] [rbp-1C8h]
  _BYTE *v69; // [rsp+1B0h] [rbp-1C0h]
  __int16 v70; // [rsp+1B8h] [rbp-1B8h]
  _BYTE v71[32]; // [rsp+1C0h] [rbp-1B0h] BYREF
  _BYTE v72[64]; // [rsp+1E0h] [rbp-190h] BYREF
  __int64 v73; // [rsp+220h] [rbp-150h]
  __int64 v74; // [rsp+228h] [rbp-148h]
  _BYTE *v75; // [rsp+230h] [rbp-140h]
  __int16 v76; // [rsp+238h] [rbp-138h]
  _BYTE v77[32]; // [rsp+240h] [rbp-130h] BYREF
  _BYTE v78[64]; // [rsp+260h] [rbp-110h] BYREF
  __int64 v79; // [rsp+2A0h] [rbp-D0h]
  __int64 v80; // [rsp+2A8h] [rbp-C8h]
  _BYTE *v81; // [rsp+2B0h] [rbp-C0h]
  __int16 v82; // [rsp+2B8h] [rbp-B8h]
  _BYTE v83[32]; // [rsp+2C0h] [rbp-B0h] BYREF
  _BYTE v84[64]; // [rsp+2E0h] [rbp-90h] BYREF
  __int64 v85; // [rsp+320h] [rbp-50h]
  __int64 v86; // [rsp+328h] [rbp-48h]
  _BYTE *v87; // [rsp+330h] [rbp-40h]
  __int16 v88; // [rsp+338h] [rbp-38h]

  v6 = a2 + 120;
  v8 = v60;
  v9 = v61;
  sub_C8CD80((__int64)v60, (__int64)v61, v6, a4, a5, a6);
  v10 = a2;
  v62 = 0;
  v63 = 0;
  v11 = *(_QWORD *)(a2 + 224);
  v12 = *(_QWORD *)(a2 + 216);
  v64 = 0;
  v13 = v11 - v12;
  if ( v11 == v12 )
  {
    v14 = 0;
  }
  else
  {
    if ( v13 > 0x7FFFFFFFFFFFFFE0LL )
      goto LABEL_24;
    v14 = sub_22077B0(v11 - v12);
    v10 = a2;
    v11 = *(_QWORD *)(a2 + 224);
    v12 = *(_QWORD *)(a2 + 216);
  }
  v62 = v14;
  v63 = v14;
  v64 = v14 + v13;
  if ( v11 == v12 )
  {
    v15 = v14;
  }
  else
  {
    v15 = v14 + v11 - v12;
    do
    {
      if ( v14 )
      {
        v16 = *(_QWORD *)v12;
        *(_BYTE *)(v14 + 24) = 0;
        *(_QWORD *)v14 = v16;
        if ( *(_BYTE *)(v12 + 24) )
        {
          *(_QWORD *)(v14 + 8) = *(_QWORD *)(v12 + 8);
          v17 = *(_QWORD *)(v12 + 16);
          *(_BYTE *)(v14 + 24) = 1;
          *(_QWORD *)(v14 + 16) = v17;
        }
      }
      v14 += 32;
      v12 += 32;
    }
    while ( v14 != v15 );
  }
  v63 = v15;
  v52 = v10;
  sub_C8CF70((__int64)v77, v78, 8, (__int64)v61, (__int64)v60);
  v18 = v62;
  v62 = 0;
  v79 = v18;
  v19 = v63;
  v63 = 0;
  v80 = v19;
  v20 = v64;
  v64 = 0;
  v81 = (_BYTE *)v20;
  sub_C8CF70((__int64)v83, v84, 8, (__int64)v78, (__int64)v77);
  v21 = v79;
  v79 = 0;
  v85 = v21;
  v22 = v80;
  v80 = 0;
  v86 = v22;
  v23 = v81;
  v81 = 0;
  v87 = v23;
  sub_C8CF70((__int64)v71, v72, 8, (__int64)v84, (__int64)v83);
  v24 = v85;
  v85 = 0;
  v73 = v24;
  v25 = v86;
  v86 = 0;
  v74 = v25;
  v26 = v87;
  v87 = 0;
  v75 = v26;
  sub_2AB1B50((__int64)v83);
  v76 = 256;
  sub_2AB1B50((__int64)v77);
  v51 = v52;
  sub_C8CD80((__int64)v55, (__int64)v56, v52, (__int64)v56, v27, v28);
  v57 = 0;
  v58 = 0;
  v9 = *(_BYTE **)(v52 + 104);
  v12 = *(_QWORD *)(v52 + 96);
  v59 = 0;
  v8 = &v9[-v12];
  if ( v9 != (_BYTE *)v12 )
  {
    if ( (unsigned __int64)v8 <= 0x7FFFFFFFFFFFFFE0LL )
    {
      v53 = &v9[-v12];
      v29 = sub_22077B0((unsigned __int64)v8);
      v8 = v53;
      v9 = *(_BYTE **)(v51 + 104);
      v12 = *(_QWORD *)(v51 + 96);
      goto LABEL_13;
    }
LABEL_24:
    sub_4261EA(v8, v9, v12);
  }
  v29 = 0;
LABEL_13:
  v57 = v29;
  v58 = v29;
  v59 = &v8[v29];
  if ( (_BYTE *)v12 == v9 )
  {
    v30 = (_BYTE *)v29;
  }
  else
  {
    v30 = &v9[v29 - v12];
    do
    {
      if ( v29 )
      {
        v31 = *(_QWORD *)v12;
        *(_BYTE *)(v29 + 24) = 0;
        *(_QWORD *)v29 = v31;
        if ( *(_BYTE *)(v12 + 24) )
        {
          *(_QWORD *)(v29 + 8) = *(_QWORD *)(v12 + 8);
          v32 = *(_QWORD *)(v12 + 16);
          *(_BYTE *)(v29 + 24) = 1;
          *(_QWORD *)(v29 + 16) = v32;
        }
      }
      v29 += 32;
      v12 += 32;
    }
    while ( (_BYTE *)v29 != v30 );
  }
  v58 = (__int64)v30;
  sub_C8CF70((__int64)v77, v78, 8, (__int64)v56, (__int64)v55);
  v33 = v57;
  v57 = 0;
  v79 = v33;
  v34 = v58;
  v58 = 0;
  v80 = v34;
  v35 = v59;
  v59 = 0;
  v81 = v35;
  sub_C8CF70((__int64)v83, v84, 8, (__int64)v78, (__int64)v77);
  v36 = v79;
  v79 = 0;
  v85 = v36;
  v86 = v80;
  v80 = 0;
  v87 = v81;
  v81 = 0;
  sub_C8CF70((__int64)v65, v66, 8, (__int64)v84, (__int64)v83);
  v37 = v85;
  v85 = 0;
  v67 = v37;
  v38 = v86;
  v86 = 0;
  v68 = v38;
  v39 = v87;
  v87 = 0;
  v69 = v39;
  sub_2AB1B50((__int64)v83);
  v70 = 256;
  sub_2AB1B50((__int64)v77);
  sub_C8CF70((__int64)v83, v84, 8, (__int64)v72, (__int64)v71);
  v40 = v73;
  v73 = 0;
  v85 = v40;
  v86 = v74;
  v74 = 0;
  v87 = v75;
  v75 = 0;
  v88 = v76;
  sub_C8CF70((__int64)v77, v78, 8, (__int64)v66, (__int64)v65);
  v41 = v67;
  v67 = 0;
  v79 = v41;
  v42 = v68;
  v68 = 0;
  v80 = v42;
  v43 = v69;
  v69 = 0;
  v81 = v43;
  v82 = v70;
  sub_C8CF70(a1, (void *)(a1 + 32), 8, (__int64)v78, (__int64)v77);
  v44 = v79;
  v79 = 0;
  *(_QWORD *)(a1 + 96) = v44;
  v45 = v80;
  v80 = 0;
  *(_QWORD *)(a1 + 104) = v45;
  v46 = v81;
  v81 = 0;
  *(_QWORD *)(a1 + 112) = v46;
  *(_WORD *)(a1 + 120) = v82;
  sub_C8CF70(a1 + 128, (void *)(a1 + 160), 8, (__int64)v84, (__int64)v83);
  v47 = v85;
  v85 = 0;
  *(_QWORD *)(a1 + 224) = v47;
  v48 = v86;
  v86 = 0;
  *(_QWORD *)(a1 + 232) = v48;
  v49 = v87;
  v87 = 0;
  *(_QWORD *)(a1 + 240) = v49;
  *(_WORD *)(a1 + 248) = v88;
  sub_2AB1B50((__int64)v77);
  sub_2AB1B50((__int64)v83);
  sub_2AB1B50((__int64)v65);
  sub_2AB1B50((__int64)v55);
  sub_2AB1B50((__int64)v71);
  sub_2AB1B50((__int64)v60);
  return a1;
}
