// Function: sub_19D1CA0
// Address: 0x19d1ca0
//
__int64 __fastcall sub_19D1CA0(__int64 a1, __int64 a2, unsigned int a3)
{
  unsigned __int64 v4; // r14
  __int64 v6; // r12
  __int64 v7; // rax
  __int64 v8; // r15
  __int64 v9; // r13
  __int64 v10; // rsi
  unsigned __int64 v11; // rcx
  __int64 v12; // rax
  __int64 v13; // rdi
  __int64 v14; // r8
  __int64 v15; // rax
  unsigned __int64 v16; // r11
  __int64 v18; // rax
  __int64 v19; // rdi
  int v20; // eax
  __int64 v21; // rax
  unsigned __int64 v22; // rax
  unsigned __int64 v23; // rdi
  unsigned int v24; // eax
  __int64 v25; // rdx
  unsigned int v26; // eax
  unsigned int v27; // r10d
  __int64 v28; // r11
  int v29; // edx
  __int64 v30; // rax
  unsigned int v31; // eax
  __int64 v32; // rax
  __int64 v33; // rdx
  int v34; // eax
  __int64 v35; // r12
  __int64 v36; // rbx
  __int64 v37; // rax
  __int64 v38; // r12
  __int64 v39; // rax
  __int64 v40; // r13
  __int64 v41; // rbx
  _QWORD *v42; // rax
  _QWORD *v43; // rax
  unsigned int v44; // eax
  __int64 v45; // rsi
  __int64 v46; // r8
  unsigned __int64 v47; // r9
  int v48; // eax
  __int64 v49; // rax
  int v50; // eax
  unsigned __int64 v51; // rax
  unsigned int v52; // esi
  int v53; // eax
  __int64 v54; // rax
  __int64 v55; // rax
  __int64 v56; // [rsp+0h] [rbp-A0h]
  unsigned __int64 v57; // [rsp+8h] [rbp-98h]
  __int64 v58; // [rsp+10h] [rbp-90h]
  __int64 v59; // [rsp+10h] [rbp-90h]
  int v60; // [rsp+18h] [rbp-88h]
  unsigned __int64 v61; // [rsp+18h] [rbp-88h]
  __int64 v62; // [rsp+18h] [rbp-88h]
  __int64 v63; // [rsp+18h] [rbp-88h]
  __int64 v64; // [rsp+18h] [rbp-88h]
  unsigned __int64 v65; // [rsp+20h] [rbp-80h]
  __int64 v66; // [rsp+20h] [rbp-80h]
  unsigned int v67; // [rsp+20h] [rbp-80h]
  __int64 v68; // [rsp+20h] [rbp-80h]
  __int64 v69; // [rsp+20h] [rbp-80h]
  unsigned __int64 v70; // [rsp+20h] [rbp-80h]
  unsigned __int64 v71; // [rsp+20h] [rbp-80h]
  __int64 v72; // [rsp+28h] [rbp-78h]
  unsigned __int64 v73; // [rsp+28h] [rbp-78h]
  __int64 v74; // [rsp+28h] [rbp-78h]
  unsigned __int64 v76; // [rsp+38h] [rbp-68h]
  unsigned __int64 v77; // [rsp+38h] [rbp-68h]
  unsigned int v78; // [rsp+38h] [rbp-68h]
  __int64 v79; // [rsp+38h] [rbp-68h]
  unsigned __int64 v80; // [rsp+38h] [rbp-68h]
  __int64 v81; // [rsp+38h] [rbp-68h]
  unsigned __int64 v82; // [rsp+38h] [rbp-68h]
  unsigned __int64 v83; // [rsp+38h] [rbp-68h]
  __m128i v84; // [rsp+40h] [rbp-60h] BYREF
  __int64 v85; // [rsp+50h] [rbp-50h]
  __int64 v86; // [rsp+58h] [rbp-48h]
  __int64 v87; // [rsp+60h] [rbp-40h]

  v4 = a2 & 0xFFFFFFFFFFFFFFF8LL;
  v6 = sub_1632FA0(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)((a2 & 0xFFFFFFFFFFFFFFF8LL) + 40) + 56LL) + 40LL));
  v7 = a3;
  v8 = 1;
  v9 = *(_QWORD *)(v4 + 24 * (v7 - (*(_DWORD *)(v4 + 20) & 0xFFFFFFF)));
  v10 = *(_QWORD *)(*(_QWORD *)v9 + 24LL);
  v11 = (unsigned int)sub_15A9FE0(v6, v10);
  while ( 2 )
  {
    switch ( *(_BYTE *)(v10 + 8) )
    {
      case 0:
      case 8:
      case 0xA:
      case 0xC:
      case 0x10:
        v49 = *(_QWORD *)(v10 + 32);
        v10 = *(_QWORD *)(v10 + 24);
        v8 *= v49;
        continue;
      case 1:
        v12 = 16;
        break;
      case 2:
        v12 = 32;
        break;
      case 3:
      case 9:
        v12 = 64;
        break;
      case 4:
        v12 = 80;
        break;
      case 5:
      case 6:
        v12 = 128;
        break;
      case 7:
        v83 = v11;
        v50 = sub_15A9520(v6, 0);
        v11 = v83;
        v12 = (unsigned int)(8 * v50);
        break;
      case 0xB:
        v12 = *(_DWORD *)(v10 + 8) >> 8;
        break;
      case 0xD:
        v80 = v11;
        v43 = (_QWORD *)sub_15A9930(v6, v10);
        v11 = v80;
        v12 = 8LL * *v43;
        break;
      case 0xE:
        v69 = v11;
        v72 = *(_QWORD *)(v10 + 24);
        v81 = *(_QWORD *)(v10 + 32);
        v44 = sub_15A9FE0(v6, v72);
        v11 = v69;
        v45 = v72;
        v46 = 1;
        v47 = v44;
        while ( 2 )
        {
          switch ( *(_BYTE *)(v45 + 8) )
          {
            case 0:
            case 8:
            case 0xA:
            case 0xC:
            case 0x10:
              v55 = *(_QWORD *)(v45 + 32);
              v45 = *(_QWORD *)(v45 + 24);
              v46 *= v55;
              continue;
            case 1:
              v51 = 16;
              goto LABEL_47;
            case 2:
              v51 = 32;
              goto LABEL_47;
            case 3:
            case 9:
              v51 = 64;
              goto LABEL_47;
            case 4:
              v51 = 80;
              goto LABEL_47;
            case 5:
            case 6:
              v51 = 128;
              goto LABEL_47;
            case 7:
              v63 = v46;
              v52 = 0;
              v70 = v47;
              v73 = v11;
              goto LABEL_52;
            case 0xB:
              v51 = *(_DWORD *)(v45 + 8) >> 8;
              goto LABEL_47;
            case 0xD:
              JUMPOUT(0x19D22F0);
            case 0xE:
              v56 = v46;
              v57 = v47;
              v59 = v69;
              v64 = *(_QWORD *)(v45 + 24);
              v74 = *(_QWORD *)(v45 + 32);
              v71 = (unsigned int)sub_15A9FE0(v6, v64);
              v54 = sub_127FA20(v6, v64);
              v11 = v59;
              v47 = v57;
              v46 = v56;
              v51 = 8 * v74 * v71 * ((v71 + ((unsigned __int64)(v54 + 7) >> 3) - 1) / v71);
              goto LABEL_47;
            case 0xF:
              v63 = v46;
              v70 = v47;
              v73 = v11;
              v52 = *(_DWORD *)(v45 + 8) >> 8;
LABEL_52:
              v53 = sub_15A9520(v6, v52);
              v11 = v73;
              v47 = v70;
              v46 = v63;
              v51 = (unsigned int)(8 * v53);
LABEL_47:
              v12 = 8 * v81 * v47 * ((v47 + ((v51 * v46 + 7) >> 3) - 1) / v47);
              break;
          }
          break;
        }
        break;
      case 0xF:
        v82 = v11;
        v48 = sub_15A9520(v6, *(_DWORD *)(v10 + 8) >> 8);
        v11 = v82;
        v12 = (unsigned int)(8 * v48);
        break;
    }
    break;
  }
  v13 = *(_QWORD *)a1;
  v14 = *(_QWORD *)(v4 + 40);
  v84.m128i_i64[0] = v9;
  v85 = 0;
  v86 = 0;
  v87 = 0;
  v76 = v11 * ((v11 + ((unsigned __int64)(v12 * v8 + 7) >> 3) - 1) / v11);
  v84.m128i_i64[1] = v76;
  v15 = sub_141C340(v13, &v84, 1u, (_QWORD *)(v4 + 24), v14, 0, 0, 0);
  if ( (v15 & 7) != 1 )
    return 0;
  v16 = v15 & 0xFFFFFFFFFFFFFFF8LL;
  if ( *(_BYTE *)((v15 & 0xFFFFFFFFFFFFFFF8LL) + 16) != 78 )
    return 0;
  v18 = *(_QWORD *)(v16 - 24);
  if ( *(_BYTE *)(v18 + 16) || (*(_BYTE *)(v18 + 33) & 0x20) == 0 || *(_DWORD *)(v18 + 36) != 133 )
    return 0;
  v19 = *(_QWORD *)(v16 + 24 * (3LL - (*(_DWORD *)(v16 + 20) & 0xFFFFFFF)));
  if ( *(_DWORD *)(v19 + 32) <= 0x40u )
  {
    if ( !*(_QWORD *)(v19 + 24) )
      goto LABEL_13;
    return 0;
  }
  v60 = *(_DWORD *)(v19 + 32);
  v65 = v16;
  v20 = sub_16A57B0(v19 + 24);
  v16 = v65;
  if ( v60 != v20 )
    return 0;
LABEL_13:
  v61 = v16;
  v66 = sub_1649C60(v9);
  if ( v66 != sub_1649C60(*(_QWORD *)(v61 - 24LL * (*(_DWORD *)(v61 + 20) & 0xFFFFFFF))) )
    return 0;
  v21 = *(_QWORD *)(v61 + 24 * (2LL - (*(_DWORD *)(v61 + 20) & 0xFFFFFFF)));
  if ( *(_BYTE *)(v21 + 16) != 13 )
    return 0;
  v22 = *(_DWORD *)(v21 + 32) <= 0x40u ? *(_QWORD *)(v21 + 24) : **(_QWORD **)(v21 + 24);
  if ( v22 < v76 )
    return 0;
  v23 = v4 + 56;
  v24 = sub_15603A0((_QWORD *)(v4 + 56), a3);
  if ( !v24 )
    return 0;
  if ( !*(_QWORD *)(a1 + 64)
    || (v67 = v24,
        v23 = a1 + 48,
        v77 = v61,
        v58 = (*(__int64 (__fastcall **)(__int64))(a1 + 72))(a1 + 48),
        !*(_QWORD *)(a1 + 96)) )
  {
    sub_4263D6(v23, a3, v25);
  }
  v62 = (*(__int64 (__fastcall **)(__int64))(a1 + 104))(a1 + 80);
  v26 = sub_15603A0((_QWORD *)(v77 + 56), 1);
  v27 = v67;
  v28 = v77;
  if ( v26 < v67 )
  {
    v29 = *(_DWORD *)(v77 + 20);
    v68 = v77;
    v78 = v27;
    v30 = sub_1649C60(*(_QWORD *)(v28 + 24 * (1LL - (v29 & 0xFFFFFFF))));
    v31 = sub_1AE99B0(v30, v78, v6, v4, v58, v62);
    v28 = v68;
    if ( v31 < v78 )
      return 0;
  }
  v79 = v28;
  v32 = *(_QWORD *)sub_1649C60(*(_QWORD *)(v28 + 24 * (1LL - (*(_DWORD *)(v28 + 20) & 0xFFFFFFF))));
  if ( *(_BYTE *)(v32 + 8) == 16 )
    v32 = **(_QWORD **)(v32 + 16);
  v33 = *(_QWORD *)v9;
  v34 = *(_DWORD *)(v32 + 8) >> 8;
  if ( *(_BYTE *)(*(_QWORD *)v9 + 8LL) == 16 )
    v33 = **(_QWORD **)(v33 + 16);
  if ( *(_DWORD *)(v33 + 8) >> 8 != v34 )
    return 0;
  v35 = *(_QWORD *)a1;
  v36 = *(_QWORD *)(v79 + 40);
  sub_141F730(&v84, v79);
  v37 = sub_141C340(v35, &v84, 0, (_QWORD *)(v4 + 24), v36, 0, 0, 0);
  if ( (v37 & 7) != 1 || v79 != (v37 & 0xFFFFFFFFFFFFFFF8LL) )
    return 0;
  v38 = sub_19D1C80(v79);
  if ( *(_QWORD *)v9 != *(_QWORD *)sub_19D1C80(v79) )
  {
    v39 = sub_19D1C80(v79);
    v40 = *(_QWORD *)v9;
    v41 = v39;
    v84.m128i_i64[0] = (__int64)"tmpcast";
    LOWORD(v85) = 259;
    v42 = sub_1648A60(56, 1u);
    v38 = (__int64)v42;
    if ( v42 )
      sub_15FD590((__int64)v42, v41, v40, (__int64)&v84, v4);
  }
  sub_19CF7F0(v4, a3, v38);
  return 1;
}
