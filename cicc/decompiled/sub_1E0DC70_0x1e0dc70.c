// Function: sub_1E0DC70
// Address: 0x1e0dc70
//
__int64 __fastcall sub_1E0DC70(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v3; // r11
  __m128i *v5; // rsi
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // r10
  __int64 v9; // rbx
  unsigned int v11; // r15d
  __int64 v12; // rax
  int v13; // edx
  __int64 v14; // r13
  __int64 v15; // rsi
  __m128i *v16; // rsi
  __int64 v18; // rax
  _QWORD *v19; // rax
  __int64 v20; // rax
  _BYTE *v21; // r11
  __int64 v22; // rdx
  _BYTE *v23; // rcx
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // r9
  unsigned __int64 v27; // rcx
  unsigned int v28; // eax
  __int64 v29; // r11
  unsigned int v30; // esi
  unsigned int v31; // eax
  __int64 v32; // rsi
  __int64 v33; // rdx
  _BYTE *v34; // rcx
  __int64 v35; // rax
  unsigned __int64 v36; // rsi
  __int64 v37; // rax
  char v38; // di
  __int64 v39; // rax
  __int64 v40; // rdi
  __int64 v41; // rax
  __int64 v42; // r11
  __int64 v43; // rax
  unsigned int v44; // esi
  __int64 v45; // rdi
  __int64 v46; // rax
  unsigned __int64 v47; // [rsp+0h] [rbp-A0h]
  __int64 v48; // [rsp+8h] [rbp-98h]
  unsigned __int64 v49; // [rsp+10h] [rbp-90h]
  unsigned __int64 v50; // [rsp+10h] [rbp-90h]
  __int64 v51; // [rsp+10h] [rbp-90h]
  __int64 v52; // [rsp+18h] [rbp-88h]
  __int64 v53; // [rsp+18h] [rbp-88h]
  unsigned __int64 v54; // [rsp+18h] [rbp-88h]
  __int64 v55; // [rsp+18h] [rbp-88h]
  __int64 v56; // [rsp+20h] [rbp-80h]
  __int64 *v57; // [rsp+20h] [rbp-80h]
  __int64 v58; // [rsp+20h] [rbp-80h]
  __int64 v59; // [rsp+20h] [rbp-80h]
  __int64 v60; // [rsp+28h] [rbp-78h]
  __int64 v61; // [rsp+28h] [rbp-78h]
  __int64 v62; // [rsp+28h] [rbp-78h]
  __int64 v63; // [rsp+28h] [rbp-78h]
  __int64 v64; // [rsp+28h] [rbp-78h]
  __int64 v65; // [rsp+30h] [rbp-70h]
  unsigned __int64 v66; // [rsp+30h] [rbp-70h]
  __int64 v67; // [rsp+30h] [rbp-70h]
  __int64 v68; // [rsp+30h] [rbp-70h]
  __int64 v69; // [rsp+30h] [rbp-70h]
  unsigned __int64 v70; // [rsp+30h] [rbp-70h]
  __int64 v71; // [rsp+38h] [rbp-68h]
  __int64 v72; // [rsp+38h] [rbp-68h]
  __int64 v73; // [rsp+38h] [rbp-68h]
  __int64 v74; // [rsp+38h] [rbp-68h]
  unsigned __int64 v75; // [rsp+38h] [rbp-68h]
  __int64 v76; // [rsp+38h] [rbp-68h]
  __int64 v77; // [rsp+40h] [rbp-60h]
  __int64 v78; // [rsp+40h] [rbp-60h]
  __int64 v79; // [rsp+40h] [rbp-60h]
  unsigned __int64 v80; // [rsp+40h] [rbp-60h]
  __int64 v81; // [rsp+40h] [rbp-60h]
  int v82; // [rsp+48h] [rbp-58h]
  _BYTE *v83; // [rsp+48h] [rbp-58h]
  __int64 v84; // [rsp+48h] [rbp-58h]
  __int64 v85; // [rsp+48h] [rbp-58h]
  __int64 v86; // [rsp+48h] [rbp-58h]
  __int64 v87; // [rsp+48h] [rbp-58h]
  _BYTE *v88; // [rsp+48h] [rbp-58h]
  __int64 v89; // [rsp+48h] [rbp-58h]
  _BYTE *v90; // [rsp+50h] [rbp-50h]
  __int64 v91; // [rsp+50h] [rbp-50h]
  __int64 v92; // [rsp+50h] [rbp-50h]
  __int64 v93; // [rsp+50h] [rbp-50h]
  __int64 v94; // [rsp+50h] [rbp-50h]
  __int64 v95; // [rsp+50h] [rbp-50h]
  __int64 v96; // [rsp+50h] [rbp-50h]
  __int64 v98; // [rsp+58h] [rbp-48h]
  __m128i v99; // [rsp+60h] [rbp-40h] BYREF

  v3 = a1;
  if ( *(_DWORD *)a1 < a3 )
    *(_DWORD *)a1 = a3;
  v5 = *(__m128i **)(a1 + 16);
  v6 = *(_QWORD *)(a1 + 8);
  v7 = ((__int64)v5->m128i_i64 - v6) >> 4;
  if ( (_DWORD)v7 )
  {
    v8 = (unsigned int)v7;
    v9 = 0;
    while ( 1 )
    {
      v11 = v9;
      v12 = 16 * v9 + v6;
      v13 = *(_DWORD *)(v12 + 8);
      if ( v13 >= 0 )
      {
        v14 = *(_QWORD *)v12;
        if ( *(_QWORD *)v12 == a2 )
          goto LABEL_31;
        v15 = *(_QWORD *)v14;
        if ( *(_QWORD *)v14 != *(_QWORD *)a2
          && (unsigned __int8)(*(_BYTE *)(v15 + 8) - 13) > 1u
          && (unsigned __int8)(*(_BYTE *)(*(_QWORD *)a2 + 8LL) - 13) > 1u )
        {
          break;
        }
      }
LABEL_5:
      if ( v8 == ++v9 )
      {
        v5 = *(__m128i **)(a1 + 16);
        v3 = a1;
        goto LABEL_14;
      }
      v6 = *(_QWORD *)(a1 + 8);
    }
    v60 = v8;
    v65 = a2;
    v72 = *(_QWORD *)(v15 + 24);
    v78 = *(_QWORD *)(a1 + 64);
    v85 = *(_QWORD *)(v15 + 32);
    v28 = sub_15A9FE0(v78, v72);
    v29 = v78;
    v40 = v78;
    v50 = v28;
    v53 = v60;
    v57 = (__int64 *)v65;
    v81 = *(_QWORD *)(v72 + 32);
    v62 = *(_QWORD *)(v72 + 24);
    v68 = v29;
    v75 = (unsigned int)sub_15A9FE0(v40, v62);
    v41 = sub_127FA20(v68, v62);
    v42 = v68;
    a2 = (__int64)v57;
    v8 = v53;
    v36 = 8
        * v50
        * v85
        * ((v50 + ((8 * v75 * v81 * ((v75 + ((unsigned __int64)(v41 + 7) >> 3) - 1) / v75) + 7) >> 3) - 1)
         / v50);
    v37 = *v57;
    v38 = *(_BYTE *)(*v57 + 8);
    v26 = 1;
    v27 = (v36 + 7) >> 3;
    while ( 2 )
    {
      switch ( v38 )
      {
        case 1:
          v18 = 16;
          goto LABEL_21;
        case 2:
          v18 = 32;
          goto LABEL_21;
        case 3:
        case 9:
          v18 = 64;
          goto LABEL_21;
        case 4:
          v18 = 80;
          goto LABEL_21;
        case 5:
        case 6:
          v18 = 128;
          goto LABEL_21;
        case 7:
          v66 = (v36 + 7) >> 3;
          v30 = 0;
          v73 = v26;
          v79 = v53;
          v86 = (__int64)v57;
          goto LABEL_37;
        case 11:
          v18 = *(_DWORD *)(v37 + 8) >> 8;
          goto LABEL_21;
        case 13:
          v66 = (v36 + 7) >> 3;
          v73 = v26;
          v79 = v53;
          v86 = (__int64)v57;
          v93 = v42;
          v18 = 8LL * *(_QWORD *)sub_15A9930(v42, v37);
          goto LABEL_38;
        case 14:
          v56 = v53;
          v49 = (v36 + 7) >> 3;
          v87 = *(_QWORD *)(v37 + 32);
          v52 = v26;
          v61 = a2;
          v67 = *(_QWORD *)(v37 + 24);
          v74 = v42;
          v31 = sub_15A9FE0(v42, v67);
          v42 = v74;
          v27 = v49;
          v94 = 1;
          v32 = v67;
          v26 = v52;
          v8 = v56;
          a2 = v61;
          v80 = v31;
          while ( 2 )
          {
            switch ( *(_BYTE *)(v32 + 8) )
            {
              case 1:
                v39 = 16;
                goto LABEL_51;
              case 2:
                v39 = 32;
                goto LABEL_51;
              case 3:
              case 9:
                v39 = 64;
                goto LABEL_51;
              case 4:
                v39 = 80;
                goto LABEL_51;
              case 5:
              case 6:
                v39 = 128;
                goto LABEL_51;
              case 7:
                v54 = v49;
                v44 = 0;
                v58 = v26;
                v63 = v8;
                v69 = a2;
                goto LABEL_57;
              case 0xB:
                v39 = *(_DWORD *)(v32 + 8) >> 8;
                goto LABEL_51;
              case 0xD:
                v54 = v49;
                v58 = v26;
                v63 = v8;
                v69 = a2;
                v39 = 8LL * *(_QWORD *)sub_15A9930(v74, v32);
                goto LABEL_58;
              case 0xE:
                v45 = v74;
                v47 = v49;
                v48 = v52;
                v51 = v56;
                v55 = v61;
                v76 = *(_QWORD *)(v32 + 32);
                v59 = *(_QWORD *)(v32 + 24);
                v64 = v42;
                v70 = (unsigned int)sub_15A9FE0(v45, v59);
                v46 = sub_127FA20(v64, v59);
                v42 = v64;
                a2 = v55;
                v8 = v51;
                v26 = v48;
                v27 = v47;
                v39 = 8 * v70 * v76 * ((v70 + ((unsigned __int64)(v46 + 7) >> 3) - 1) / v70);
                goto LABEL_51;
              case 0xF:
                v54 = v49;
                v58 = v26;
                v63 = v8;
                v44 = *(_DWORD *)(v32 + 8) >> 8;
                v69 = a2;
LABEL_57:
                v39 = 8 * (unsigned int)sub_15A9520(v74, v44);
LABEL_58:
                v42 = v74;
                a2 = v69;
                v8 = v63;
                v26 = v58;
                v27 = v54;
LABEL_51:
                v18 = 8 * v80 * v87 * ((v80 + ((unsigned __int64)(v94 * v39 + 7) >> 3) - 1) / v80);
                goto LABEL_21;
              case 0x10:
                v43 = v94 * *(_QWORD *)(v32 + 32);
                v32 = *(_QWORD *)(v32 + 24);
                v94 = v43;
                continue;
              default:
                goto LABEL_64;
            }
          }
        case 15:
          v66 = (v36 + 7) >> 3;
          v73 = v26;
          v79 = v53;
          v30 = *(_DWORD *)(v37 + 8) >> 8;
          v86 = (__int64)v57;
LABEL_37:
          v93 = v42;
          v18 = 8 * (unsigned int)sub_15A9520(v42, v30);
LABEL_38:
          v42 = v93;
          a2 = v86;
          v8 = v79;
          v26 = v73;
          v27 = v66;
LABEL_21:
          if ( (unsigned __int64)(v18 * v26 + 7) >> 3 != v27 || v27 > 0x80 )
            goto LABEL_5;
          v71 = a2;
          v77 = v8;
          v90 = (_BYTE *)v42;
          v82 = v27;
          v19 = (_QWORD *)sub_16498A0(v14);
          v20 = sub_1644900(v19, 8 * v82);
          v21 = v90;
          v8 = v77;
          v22 = v20;
          a2 = v71;
          if ( *(_BYTE *)(*(_QWORD *)v14 + 8LL) == 15 )
          {
            v34 = v90;
            v88 = v90;
            v95 = v20;
            v35 = sub_14D7A60(45, v14, v20, v34);
            v22 = v95;
            v21 = v88;
            v8 = v77;
            a2 = v71;
            v14 = v35;
          }
          else if ( v20 != *(_QWORD *)v14 )
          {
            v23 = v90;
            v83 = v90;
            v91 = v20;
            v24 = sub_14D7A60(47, v14, v20, v23);
            a2 = v71;
            v8 = v77;
            v21 = v83;
            v22 = v91;
            v14 = v24;
          }
          if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 15 )
          {
            v89 = v8;
            v96 = a2;
            v25 = sub_14D7A60(45, a2, v22, v21);
            a2 = v96;
            v8 = v89;
          }
          else
          {
            v25 = a2;
            if ( v22 != *(_QWORD *)a2 )
            {
              v84 = v8;
              v92 = a2;
              v25 = sub_14D7A60(47, a2, v22, v21);
              v8 = v84;
              a2 = v92;
            }
          }
          if ( v14 != v25 )
            goto LABEL_5;
          v12 = 16 * v9 + *(_QWORD *)(a1 + 8);
          v13 = *(_DWORD *)(v12 + 8);
          break;
        case 16:
          v33 = *(_QWORD *)(v37 + 32);
          v37 = *(_QWORD *)(v37 + 24);
          v26 *= v33;
          v38 = *(_BYTE *)(v37 + 8);
          continue;
        default:
LABEL_64:
          ++*(_DWORD *)(16 * v9 + 0x1C0);
          BUG();
      }
      break;
    }
LABEL_31:
    if ( (v13 & 0x7FFFFFFFu) < a3 )
      *(_DWORD *)(v12 + 8) = a3;
  }
  else
  {
LABEL_14:
    v99.m128i_i64[0] = a2;
    v99.m128i_i32[2] = a3;
    if ( *(__m128i **)(v3 + 24) == v5 )
    {
      v98 = v3;
      sub_1E0DAF0((const __m128i **)(v3 + 8), v5, &v99);
      v3 = v98;
      v16 = *(__m128i **)(v98 + 16);
    }
    else
    {
      if ( v5 )
      {
        *v5 = _mm_loadu_si128(&v99);
        v5 = *(__m128i **)(v3 + 16);
      }
      v16 = v5 + 1;
      *(_QWORD *)(v3 + 16) = v16;
    }
    return (unsigned int)(((__int64)v16->m128i_i64 - *(_QWORD *)(v3 + 8)) >> 4) - 1;
  }
  return v11;
}
