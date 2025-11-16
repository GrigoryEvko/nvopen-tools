// Function: sub_30F2600
// Address: 0x30f2600
//
__int64 __fastcall sub_30F2600(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // r13
  __int64 v7; // rbx
  __int64 v8; // r14
  unsigned __int8 *v9; // rdi
  unsigned __int64 v10; // rdx
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // rsi
  __int64 v14; // rcx
  unsigned __int8 *v15; // rbx
  unsigned __int8 *v16; // r12
  unsigned __int8 *v17; // r10
  unsigned __int8 *v18; // r14
  __int64 v19; // rax
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r9
  __int64 v24; // rax
  const char *v25; // rax
  _BYTE *v26; // rax
  unsigned __int8 *v27; // r10
  _BYTE *v28; // rax
  __int64 v29; // rsi
  _BYTE *v30; // rax
  unsigned __int8 *v31; // r10
  __int64 v32; // rsi
  __int64 v33; // rax
  unsigned __int8 *v34; // r10
  unsigned __int64 v35; // rax
  __int64 v36; // rsi
  __int64 v37; // rax
  unsigned __int8 *v38; // r10
  unsigned __int64 v39; // rax
  __int64 v40; // r13
  unsigned __int64 v41; // rax
  const char *v42; // rax
  _BYTE *v43; // rax
  unsigned __int8 *v44; // r10
  __int64 v45; // rax
  __int64 v46; // r13
  unsigned __int64 v47; // rax
  __int64 v48; // rax
  __int64 v49; // r13
  unsigned __int64 v50; // rax
  __int64 v51; // r13
  unsigned __int64 v52; // rax
  int v53; // eax
  int v54; // eax
  _BYTE *v55; // rax
  unsigned __int8 *v56; // r10
  _BYTE *v57; // rax
  _BYTE *v58; // rax
  __int64 v59; // rcx
  __int64 v60; // r8
  __int64 v61; // r9
  unsigned __int8 *v62; // [rsp+8h] [rbp-218h]
  unsigned __int8 *v63; // [rsp+8h] [rbp-218h]
  __int64 v64; // [rsp+10h] [rbp-210h]
  __int64 v65; // [rsp+10h] [rbp-210h]
  unsigned __int8 *v66; // [rsp+18h] [rbp-208h]
  unsigned __int8 *v67; // [rsp+18h] [rbp-208h]
  unsigned __int64 v68; // [rsp+18h] [rbp-208h]
  unsigned __int64 v69; // [rsp+18h] [rbp-208h]
  unsigned __int8 *v70; // [rsp+18h] [rbp-208h]
  unsigned __int8 *v72; // [rsp+30h] [rbp-1F0h]
  unsigned __int8 *v73; // [rsp+30h] [rbp-1F0h]
  unsigned __int8 *v74; // [rsp+30h] [rbp-1F0h]
  unsigned int v75; // [rsp+30h] [rbp-1F0h]
  unsigned __int8 *v76; // [rsp+30h] [rbp-1F0h]
  unsigned int v77; // [rsp+30h] [rbp-1F0h]
  unsigned __int64 v78; // [rsp+30h] [rbp-1F0h]
  unsigned __int64 v79; // [rsp+30h] [rbp-1F0h]
  unsigned __int8 *v80; // [rsp+30h] [rbp-1F0h]
  _BYTE *v81; // [rsp+30h] [rbp-1F0h]
  unsigned __int8 v82; // [rsp+3Ch] [rbp-1E4h]
  __int16 v83; // [rsp+3Eh] [rbp-1E2h]
  __int16 v84; // [rsp+40h] [rbp-1E0h]
  __int16 v85; // [rsp+42h] [rbp-1DEh]
  __int16 v86; // [rsp+44h] [rbp-1DCh]
  unsigned __int8 v87; // [rsp+46h] [rbp-1DAh]
  __int64 v88; // [rsp+50h] [rbp-1D0h]
  __int64 v89; // [rsp+58h] [rbp-1C8h]
  __int64 i; // [rsp+58h] [rbp-1C8h]
  __m128i v91[2]; // [rsp+60h] [rbp-1C0h] BYREF
  char v92; // [rsp+80h] [rbp-1A0h]
  char v93; // [rsp+81h] [rbp-19Fh]
  __m128i v94[2]; // [rsp+90h] [rbp-190h] BYREF
  char v95; // [rsp+B0h] [rbp-170h]
  char v96; // [rsp+B1h] [rbp-16Fh]
  __m128i v97[3]; // [rsp+C0h] [rbp-160h] BYREF
  __m128i v98[2]; // [rsp+F0h] [rbp-130h] BYREF
  char v99; // [rsp+110h] [rbp-110h]
  char v100; // [rsp+111h] [rbp-10Fh]
  __m128i v101; // [rsp+120h] [rbp-100h] BYREF
  __int64 v102; // [rsp+130h] [rbp-F0h]
  __int64 v103; // [rsp+138h] [rbp-E8h]
  _QWORD v104[4]; // [rsp+140h] [rbp-E0h] BYREF
  __int64 v105[7]; // [rsp+160h] [rbp-C0h] BYREF
  unsigned __int64 v106[2]; // [rsp+198h] [rbp-88h] BYREF
  _BYTE v107[16]; // [rsp+1A8h] [rbp-78h] BYREF
  _QWORD v108[3]; // [rsp+1B8h] [rbp-68h] BYREF
  unsigned __int64 v109; // [rsp+1D0h] [rbp-50h]
  _BYTE *v110; // [rsp+1D8h] [rbp-48h]
  __int64 v111; // [rsp+1E0h] [rbp-40h]
  unsigned __int64 *v112; // [rsp+1E8h] [rbp-38h]

  v6 = *(_QWORD *)(a3 + 40);
  v7 = sub_B2BEC0(a3);
  v8 = sub_BC1CD0(a4, &unk_4F86540, a3);
  v88 = sub_BC1CD0(a4, &unk_4F86630, a3);
  v89 = sub_BC1CD0(a4, &unk_4F81450, a3);
  v105[2] = v7;
  v105[0] = v6;
  v105[6] = sub_BC1CD0(a4, &unk_4F6D3F8, a3) + 8;
  v106[0] = (unsigned __int64)v107;
  v111 = 0x100000000LL;
  v105[5] = v89 + 8;
  v9 = (unsigned __int8 *)v108;
  v105[1] = v6 + 232;
  v108[0] = &unk_49DD210;
  v105[3] = v8 + 8;
  v105[4] = v88 + 8;
  v106[1] = 0;
  v107[0] = 0;
  v108[1] = 0;
  v108[2] = 0;
  v109 = 0;
  v110 = 0;
  v112 = v106;
  sub_CB5980((__int64)v108, 0, 0, 0);
  if ( (*(_BYTE *)(a3 + 7) & 0x10) == 0 && (*(_BYTE *)(a3 + 32) & 0xFu) - 7 > 1 )
  {
    v101.m128i_i64[0] = (__int64)"Unusual: Unnamed function with non-local linkage";
    LOWORD(v104[0]) = 259;
    sub_CA0E80((__int64)&v101, (__int64)v108);
    v57 = v110;
    if ( (unsigned __int64)v110 >= v109 )
    {
      sub_CB5D20((__int64)v108, 10);
    }
    else
    {
      ++v110;
      *v57 = 10;
    }
    if ( *(_BYTE *)a3 <= 0x1Cu )
    {
      v9 = (unsigned __int8 *)a3;
      sub_A5BF40((unsigned __int8 *)a3, (__int64)v108, 1, v105[0]);
      v58 = v110;
      if ( (unsigned __int64)v110 < v109 )
        goto LABEL_88;
    }
    else
    {
      v9 = (unsigned __int8 *)a3;
      sub_A69870(a3, v108, 0);
      v58 = v110;
      if ( (unsigned __int64)v110 < v109 )
      {
LABEL_88:
        v10 = (unsigned __int64)(v58 + 1);
        v110 = v58 + 1;
        *v58 = 10;
        goto LABEL_3;
      }
    }
    v9 = (unsigned __int8 *)v108;
    sub_CB5D20((__int64)v108, 10);
  }
LABEL_3:
  v13 = *(_QWORD *)(a3 + 80);
  for ( i = v13; a3 + 72 != i; v13 = i )
  {
    v14 = i;
    v15 = *(unsigned __int8 **)(i + 32);
    v16 = (unsigned __int8 *)(i + 24);
    i = *(_QWORD *)(i + 8);
    while ( v16 != v15 )
    {
LABEL_6:
      v17 = v15;
      v15 = (unsigned __int8 *)*((_QWORD *)v15 + 1);
      v18 = v17 - 24;
      switch ( *(v17 - 24) )
      {
        case 0x1Eu:
          v73 = v17;
          v9 = *(unsigned __int8 **)(*((_QWORD *)v17 + 2) + 72LL);
          if ( !(unsigned __int8)sub_B2D610((__int64)v9, 36) )
          {
            if ( (*((_DWORD *)v73 - 5) & 0x7FFFFFF) != 0 )
            {
              v29 = *(_QWORD *)&v73[-32 * (*((_DWORD *)v73 - 5) & 0x7FFFFFF) - 24];
              if ( v29 )
              {
                v101.m128i_i64[0] = 0;
                v9 = (unsigned __int8 *)v105;
                v101.m128i_i64[1] = (__int64)v104;
                v102 = 4;
                LODWORD(v103) = 0;
                BYTE4(v103) = 1;
                v30 = (_BYTE *)sub_30EFD90(v105, v29, 1, (__int64)&v101, v11, v12);
                v31 = v73;
                if ( !BYTE4(v103) )
                {
                  v9 = (unsigned __int8 *)v101.m128i_i64[1];
                  v70 = v73;
                  v81 = v30;
                  _libc_free(v101.m128i_u64[1]);
                  v31 = v70;
                  v30 = v81;
                }
                if ( *v30 == 60 )
                {
                  v72 = v31;
                  v25 = "Unusual: Returning alloca value";
                  BYTE1(v104[0]) = 1;
                  goto LABEL_16;
                }
              }
            }
            continue;
          }
          BYTE1(v104[0]) = 1;
          v42 = "Unusual: Return statement in function with noreturn attribute";
LABEL_46:
          v101.m128i_i64[0] = (__int64)v42;
          LOBYTE(v104[0]) = 3;
          sub_CA0E80((__int64)&v101, (__int64)v108);
          v43 = v110;
          v44 = v73;
          if ( (unsigned __int64)v110 >= v109 )
          {
            sub_CB5D20((__int64)v108, 10);
            v44 = v73;
          }
          else
          {
            ++v110;
            *v43 = 10;
          }
          if ( *(v44 - 24) > 0x1Cu )
            goto LABEL_68;
          goto LABEL_49;
        case 0x1Fu:
        case 0x20u:
        case 0x23u:
        case 0x25u:
        case 0x26u:
        case 0x27u:
        case 0x29u:
        case 0x2Au:
        case 0x2Bu:
        case 0x2Du:
        case 0x2Eu:
        case 0x2Fu:
        case 0x32u:
        case 0x35u:
        case 0x39u:
        case 0x3Au:
        case 0x3Fu:
        case 0x40u:
        case 0x43u:
        case 0x44u:
        case 0x45u:
        case 0x46u:
        case 0x47u:
        case 0x48u:
        case 0x49u:
        case 0x4Au:
        case 0x4Bu:
        case 0x4Cu:
        case 0x4Du:
        case 0x4Eu:
        case 0x4Fu:
        case 0x50u:
        case 0x51u:
        case 0x52u:
        case 0x53u:
        case 0x54u:
        case 0x56u:
        case 0x57u:
        case 0x58u:
        case 0x5Cu:
        case 0x5Du:
        case 0x5Eu:
        case 0x5Fu:
        case 0x60u:
          continue;
        case 0x21u:
          v9 = (unsigned __int8 *)v105;
          v72 = v17;
          v24 = **((_QWORD **)v17 - 4);
          v102 = 0;
          v103 = 0;
          v101.m128i_i64[0] = v24;
          v101.m128i_i64[1] = 0xBFFFFFFFFFFFFFFELL;
          v104[0] = 0;
          v104[1] = 0;
          sub_30F09A0((__int64)v105, v17 - 24, v101.m128i_i64, v82, 0, 8);
          if ( (*((_DWORD *)v72 - 5) & 0x7FFFFFF) != 1 )
            continue;
          BYTE1(v104[0]) = 1;
          v25 = "Undefined behavior: indirectbr with no destinations";
          goto LABEL_16;
        case 0x22u:
        case 0x28u:
        case 0x55u:
          v9 = (unsigned __int8 *)v105;
          sub_30F1370(v105, v17 - 24);
          if ( v16 == v15 )
            goto LABEL_8;
          goto LABEL_6;
        case 0x24u:
          v45 = *(_QWORD *)(*((_QWORD *)v17 + 2) + 56LL);
          if ( v45 && v18 == (unsigned __int8 *)(v45 - 24) )
            continue;
          v73 = v17;
          v9 = (unsigned __int8 *)((*(_QWORD *)v17 & 0xFFFFFFFFFFFFFFF8LL) - 24);
          if ( (*(_QWORD *)v17 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
            v9 = 0;
          if ( (unsigned __int8)sub_B46970(v9) )
            continue;
          BYTE1(v104[0]) = 1;
          v42 = "Unusual: unreachable immediately preceded by instruction without side effects";
          goto LABEL_46;
        case 0x2Cu:
          if ( (unsigned int)**((unsigned __int8 **)v17 - 11) - 12 > 1 )
            continue;
          v73 = v17;
          if ( (unsigned int)**((unsigned __int8 **)v17 - 7) - 12 > 1 )
            continue;
          BYTE1(v104[0]) = 1;
          v42 = "Undefined result: sub(undef, undef)";
          goto LABEL_46;
        case 0x30u:
        case 0x31u:
        case 0x33u:
        case 0x34u:
          v9 = (unsigned __int8 *)v105;
          sub_30F0430((__int64)v105, v17 - 24);
          continue;
        case 0x36u:
        case 0x37u:
        case 0x38u:
          v9 = (unsigned __int8 *)v105;
          sub_30F0800((__int64)v105, (__int64)(v17 - 24), v10, v14, v11, v12);
          continue;
        case 0x3Bu:
          if ( (unsigned int)**((unsigned __int8 **)v17 - 11) - 12 > 1 )
            continue;
          v73 = v17;
          if ( (unsigned int)**((unsigned __int8 **)v17 - 7) - 12 > 1 )
            continue;
          BYTE1(v104[0]) = 1;
          v42 = "Undefined result: xor(undef, undef)";
          goto LABEL_46;
        case 0x3Cu:
          if ( **((_BYTE **)v17 - 7) != 17 )
            continue;
          v10 = *((_QWORD *)v17 + 2);
          v48 = *(_QWORD *)(*(_QWORD *)(v10 + 72) + 80LL);
          if ( v48 )
          {
            if ( v10 == v48 - 24 )
              continue;
          }
          v73 = v17;
          v42 = "Pessimization: Static alloca outside of entry block";
          BYTE1(v104[0]) = 1;
          goto LABEL_46;
        case 0x3Du:
          v46 = *((_QWORD *)v17 - 2);
          _BitScanReverse64(&v47, 1LL << (*((_WORD *)v17 - 11) >> 1));
          LOBYTE(v47) = 63 - (v47 ^ 0x3F);
          BYTE1(v47) = 1;
          v83 = v47;
          sub_D665A0(&v101, (__int64)(v17 - 24));
          v9 = (unsigned __int8 *)v105;
          sub_30F09A0((__int64)v105, v18, v101.m128i_i64, v83, v46, 1);
          continue;
        case 0x3Eu:
          v51 = *(_QWORD *)(*((_QWORD *)v17 - 11) + 8LL);
          _BitScanReverse64(&v52, 1LL << (*((_WORD *)v17 - 11) >> 1));
          LOBYTE(v52) = 63 - (v52 ^ 0x3F);
          BYTE1(v52) = 1;
          v84 = v52;
          sub_D66630(&v101, (__int64)(v17 - 24));
          v9 = (unsigned __int8 *)v105;
          sub_30F09A0((__int64)v105, v18, v101.m128i_i64, v84, v51, 2);
          continue;
        case 0x41u:
          v49 = *(_QWORD *)(*((_QWORD *)v17 - 15) + 8LL);
          _BitScanReverse64(&v50, 1LL << *(v17 - 21));
          LOBYTE(v50) = 63 - (v50 ^ 0x3F);
          BYTE1(v50) = 1;
          v85 = v50;
          sub_D66720(&v101, (__int64)(v17 - 24));
          v9 = (unsigned __int8 *)v105;
          sub_30F09A0((__int64)v105, v18, v101.m128i_i64, v85, v49, 2);
          continue;
        case 0x42u:
          v40 = *(_QWORD *)(*((_QWORD *)v17 - 11) + 8LL);
          _BitScanReverse64(&v41, 1LL << (*((_WORD *)v17 - 11) >> 9));
          LOBYTE(v41) = 63 - (v41 ^ 0x3F);
          BYTE1(v41) = 1;
          v86 = v41;
          sub_D667B0(&v101, (__int64)(v17 - 24));
          v9 = (unsigned __int8 *)v105;
          sub_30F09A0((__int64)v105, v18, v101.m128i_i64, v86, v40, 2);
          continue;
        case 0x59u:
          sub_D666C0(&v101, (__int64)(v17 - 24));
          v9 = (unsigned __int8 *)v105;
          sub_30F09A0((__int64)v105, v18, v101.m128i_i64, v87, 0, 3);
          continue;
        case 0x5Au:
          v36 = *((_QWORD *)v17 - 7);
          v9 = (unsigned __int8 *)v105;
          v76 = v17;
          v101.m128i_i64[0] = 0;
          v101.m128i_i64[1] = (__int64)v104;
          v102 = 4;
          LODWORD(v103) = 0;
          BYTE4(v103) = 1;
          v37 = sub_30EFD90(v105, v36, 0, (__int64)&v101, v11, v12);
          v38 = v76;
          v10 = v37;
          if ( !BYTE4(v103) )
          {
            v9 = (unsigned __int8 *)v101.m128i_i64[1];
            v67 = v76;
            v79 = v37;
            _libc_free(v101.m128i_u64[1]);
            v38 = v67;
            v10 = v79;
          }
          if ( *(_BYTE *)v10 != 17 )
            continue;
          v14 = *(_QWORD *)(*((_QWORD *)v38 - 11) + 8LL);
          if ( *(_BYTE *)(v14 + 8) == 18 )
            continue;
          v77 = *(_DWORD *)(v10 + 32);
          if ( v77 > 0x40 )
          {
            v9 = (unsigned __int8 *)(v10 + 24);
            v62 = v38;
            v64 = *(_QWORD *)(*((_QWORD *)v38 - 11) + 8LL);
            v68 = v10;
            v53 = sub_C444A0(v10 + 24);
            v14 = v64;
            v38 = v62;
            if ( v77 - v53 > 0x40 )
              goto LABEL_75;
            v39 = **(_QWORD **)(v68 + 24);
          }
          else
          {
            v39 = *(_QWORD *)(v10 + 24);
          }
          v10 = *(unsigned int *)(v14 + 32);
          if ( v10 > v39 )
            continue;
LABEL_75:
          v72 = v38;
          v25 = "Undefined result: extractelement index out of range";
          BYTE1(v104[0]) = 1;
LABEL_16:
          v101.m128i_i64[0] = (__int64)v25;
          LOBYTE(v104[0]) = 3;
          sub_CA0E80((__int64)&v101, (__int64)v108);
          v26 = v110;
          v27 = v72;
          if ( (unsigned __int64)v110 >= v109 )
          {
            sub_CB5D20((__int64)v108, 10);
            v27 = v72;
          }
          else
          {
            ++v110;
            *v26 = 10;
          }
          if ( *(v27 - 24) > 0x1Cu )
          {
LABEL_68:
            v9 = v18;
            sub_A69870((__int64)v18, v108, 0);
            v28 = v110;
            if ( (unsigned __int64)v110 < v109 )
              goto LABEL_50;
          }
          else
          {
            v9 = v18;
            sub_A5BF40(v18, (__int64)v108, 1, v105[0]);
            v28 = v110;
            if ( v109 > (unsigned __int64)v110 )
              goto LABEL_50;
          }
          goto LABEL_20;
        case 0x5Bu:
          v32 = *((_QWORD *)v17 - 7);
          v9 = (unsigned __int8 *)v105;
          v74 = v17;
          v101.m128i_i64[0] = 0;
          v101.m128i_i64[1] = (__int64)v104;
          v102 = 4;
          LODWORD(v103) = 0;
          BYTE4(v103) = 1;
          v33 = sub_30EFD90(v105, v32, 0, (__int64)&v101, v11, v12);
          v34 = v74;
          v10 = v33;
          if ( !BYTE4(v103) )
          {
            v9 = (unsigned __int8 *)v101.m128i_i64[1];
            v66 = v74;
            v78 = v33;
            _libc_free(v101.m128i_u64[1]);
            v34 = v66;
            v10 = v78;
          }
          if ( *(_BYTE *)v10 != 17 )
            continue;
          v14 = *((_QWORD *)v34 - 2);
          if ( *(_BYTE *)(v14 + 8) == 18 )
            continue;
          v75 = *(_DWORD *)(v10 + 32);
          if ( v75 > 0x40 )
          {
            v9 = (unsigned __int8 *)(v10 + 24);
            v63 = v34;
            v65 = *((_QWORD *)v34 - 2);
            v69 = v10;
            v54 = sub_C444A0(v10 + 24);
            v14 = v65;
            v34 = v63;
            if ( v75 - v54 > 0x40 )
              goto LABEL_79;
            v35 = **(_QWORD **)(v69 + 24);
          }
          else
          {
            v35 = *(_QWORD *)(v10 + 24);
          }
          v10 = *(unsigned int *)(v14 + 32);
          if ( v10 > v35 )
            continue;
LABEL_79:
          v80 = v34;
          v101.m128i_i64[0] = (__int64)"Undefined result: insertelement index out of range";
          LOWORD(v104[0]) = 259;
          sub_CA0E80((__int64)&v101, (__int64)v108);
          v55 = v110;
          v56 = v80;
          if ( (unsigned __int64)v110 >= v109 )
          {
            sub_CB5D20((__int64)v108, 10);
            v56 = v80;
          }
          else
          {
            ++v110;
            *v55 = 10;
          }
          if ( *(v56 - 24) <= 0x1Cu )
          {
LABEL_49:
            v9 = v18;
            sub_A5BF40(v18, (__int64)v108, 1, v105[0]);
            v28 = v110;
            if ( (unsigned __int64)v110 >= v109 )
              goto LABEL_20;
          }
          else
          {
            v9 = v18;
            sub_A69870((__int64)v18, v108, 0);
            v28 = v110;
            if ( v109 <= (unsigned __int64)v110 )
            {
LABEL_20:
              v9 = (unsigned __int8 *)v108;
              sub_CB5D20((__int64)v108, 10);
              continue;
            }
          }
LABEL_50:
          v10 = (unsigned __int64)(v28 + 1);
          v110 = v28 + 1;
          *v28 = 10;
          break;
        default:
          BUG();
      }
    }
LABEL_8:
    ;
  }
  v19 = sub_C5F790((__int64)v9, v13);
  sub_CB6200(v19, (unsigned __int8 *)*v112, v112[1]);
  if ( byte_50312C8 && v112[1] )
  {
    v100 = 1;
    v98[0].m128i_i64[0] = (__int64)")";
    v94[0].m128i_i64[0] = (__int64)"lint-abort-on-error";
    v91[0].m128i_i64[0] = (__int64)"Linter found errors, aborting. (enabled by --";
    v99 = 3;
    v96 = 1;
    v95 = 3;
    v93 = 1;
    v92 = 3;
    sub_9C6370(v97, v91, v94, v20, v21, v22);
    sub_9C6370(&v101, v97, v98, v59, v60, v61);
    sub_C64D30((__int64)&v101, 0);
  }
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_QWORD *)(a1 + 16) = 0x100000002LL;
  *(_QWORD *)(a1 + 64) = 2;
  *(_QWORD *)(a1 + 32) = &qword_4F82400;
  *(_DWORD *)(a1 + 72) = 0;
  *(_BYTE *)(a1 + 76) = 1;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)a1 = 1;
  v108[0] = &unk_49DD210;
  sub_CB5840((__int64)v108);
  if ( (_BYTE *)v106[0] != v107 )
    j_j___libc_free_0(v106[0]);
  return a1;
}
