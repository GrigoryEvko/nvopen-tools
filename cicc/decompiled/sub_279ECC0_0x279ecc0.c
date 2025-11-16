// Function: sub_279ECC0
// Address: 0x279ecc0
//
__int64 __fastcall sub_279ECC0(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v4; // rax
  __int64 v5; // rsi
  __int64 v6; // rcx
  __int64 v7; // rdx
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // r15
  __int64 v11; // rdi
  __int64 v12; // rcx
  int v13; // edx
  unsigned __int8 v15; // al
  int v16; // eax
  _BYTE *v17; // rbx
  __int64 v18; // rdx
  __int64 *v19; // r12
  __int64 *v20; // rax
  __int64 v21; // rax
  int v22; // r15d
  __int64 *v23; // rax
  __int64 v24; // rax
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rax
  unsigned int v30; // ebx
  unsigned int v31; // r8d
  unsigned __int8 v32; // cl
  __int64 v33; // rcx
  __int64 v34; // rsi
  __int64 v35; // rax
  __int64 *v36; // rax
  unsigned __int64 v37; // rax
  __int64 v38; // rbx
  int v39; // eax
  unsigned int v40; // r15d
  int v41; // r13d
  int v42; // esi
  __int64 *v43; // r8
  unsigned int v44; // ecx
  __int64 *v45; // rdx
  __int64 v46; // r9
  int *v47; // rax
  int v48; // edx
  __int64 v49; // rax
  unsigned int v50; // esi
  unsigned __int32 v51; // edx
  unsigned __int32 v52; // ecx
  unsigned int v53; // r8d
  __int64 *v54; // rdx
  __int8 v55; // di
  __int64 v56; // rbx
  unsigned int v57; // eax
  __int64 v58; // r14
  __int64 *v59; // r8
  int v60; // esi
  unsigned int v61; // edx
  __int64 *v62; // rax
  __int64 v63; // r10
  int v64; // eax
  __int64 v65; // rax
  __int64 v66; // r9
  __int64 v67; // rcx
  char v68; // r11
  int v69; // eax
  int v70; // r11d
  __int64 *v71; // r10
  unsigned __int8 *v72; // rax
  __int64 v73; // rdi
  __int64 v74; // rax
  __int64 v75; // r8
  __int64 v76; // r9
  __int64 v77; // rax
  int v78; // [rsp+10h] [rbp-180h]
  __int64 v79; // [rsp+18h] [rbp-178h]
  __int64 v80; // [rsp+20h] [rbp-170h]
  __int64 *v81; // [rsp+20h] [rbp-170h]
  __int64 v82; // [rsp+28h] [rbp-168h]
  unsigned int v83; // [rsp+28h] [rbp-168h]
  __int64 v84; // [rsp+38h] [rbp-158h] BYREF
  __int64 *v85; // [rsp+40h] [rbp-150h] BYREF
  __int64 v86; // [rsp+48h] [rbp-148h]
  __m128i v87; // [rsp+50h] [rbp-140h] BYREF
  __int64 *v88; // [rsp+60h] [rbp-130h] BYREF
  __int64 v89; // [rsp+68h] [rbp-128h]
  __int64 v90; // [rsp+70h] [rbp-120h]
  __int64 v91; // [rsp+78h] [rbp-118h]
  __int64 v92; // [rsp+80h] [rbp-110h]
  __int64 v93; // [rsp+88h] [rbp-108h]
  __int16 v94; // [rsp+90h] [rbp-100h]
  char v95; // [rsp+160h] [rbp-30h] BYREF

  v2 = a1;
  if ( *(_BYTE *)a2 == 85 )
  {
    v28 = *(_QWORD *)(a2 - 32);
    if ( v28 )
    {
      if ( !*(_BYTE *)v28
        && *(_QWORD *)(v28 + 24) == *(_QWORD *)(a2 + 80)
        && (*(_BYTE *)(v28 + 33) & 0x20) != 0
        && (unsigned int)(*(_DWORD *)(v28 + 36) - 68) <= 3 )
      {
        goto LABEL_29;
      }
    }
  }
  v4 = sub_B43CC0(a2);
  v5 = *(_QWORD *)(a1 + 40);
  v6 = *(_QWORD *)(a1 + 32);
  v7 = *(_QWORD *)(a1 + 24);
  v87.m128i_i64[0] = v4;
  v90 = v5;
  v87.m128i_i64[1] = v6;
  v88 = 0;
  v89 = v7;
  v91 = 0;
  v92 = 0;
  v93 = 0;
  v94 = 257;
  v10 = sub_1020E10(a2, &v87, v7, v6, v8, v9);
  if ( v10 )
  {
    if ( *(_QWORD *)(a2 + 16) )
    {
      sub_30EC4B0(*(_QWORD *)(a1 + 104), a2);
      sub_BD84D0(a2, v10);
      if ( !sub_F50EE0((unsigned __int8 *)a2, *(__int64 **)(a1 + 32)) )
        goto LABEL_5;
      goto LABEL_21;
    }
    if ( sub_F50EE0((unsigned __int8 *)a2, *(__int64 **)(a1 + 32)) )
    {
LABEL_21:
      sub_278A7A0(a1 + 136, (_BYTE *)a2);
      v27 = *(unsigned int *)(a1 + 656);
      if ( v27 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 660) )
      {
        sub_C8D5F0(a1 + 648, (const void *)(a1 + 664), v27 + 1, 8u, v25, v26);
        v27 = *(unsigned int *)(a1 + 656);
      }
      *(_QWORD *)(*(_QWORD *)(a1 + 648) + 8 * v27) = a2;
      ++*(_DWORD *)(a1 + 656);
LABEL_5:
      v11 = *(_QWORD *)(a1 + 16);
      if ( v11 )
      {
        v12 = *(_QWORD *)(v10 + 8);
        v13 = *(unsigned __int8 *)(v12 + 8);
        if ( (unsigned int)(v13 - 17) <= 1 )
          LOBYTE(v13) = *(_BYTE *)(**(_QWORD **)(v12 + 16) + 8LL);
        if ( (_BYTE)v13 == 14 )
        {
          v34 = v10;
          LODWORD(v10) = 1;
          sub_102B9D0(v11, v34);
          return (unsigned int)v10;
        }
      }
      goto LABEL_9;
    }
  }
  v15 = *(_BYTE *)a2;
  if ( *(_BYTE *)a2 == 85 )
  {
    v29 = *(_QWORD *)(a2 - 32);
    if ( v29
      && !*(_BYTE *)v29
      && *(_QWORD *)(v29 + 24) == *(_QWORD *)(a2 + 80)
      && (*(_BYTE *)(v29 + 33) & 0x20) != 0
      && *(_DWORD *)(v29 + 36) == 11 )
    {
      return sub_279CC20(a1, a2);
    }
    goto LABEL_33;
  }
  if ( v15 == 61 )
  {
    LODWORD(v10) = sub_279C520(a1, a2);
    if ( !(_BYTE)v10 )
    {
      v16 = sub_2792F80(a1 + 136, a2);
      sub_27915B0(a1 + 352, v16, a2, *(_QWORD *)(a2 + 40));
      return (unsigned int)v10;
    }
LABEL_9:
    LODWORD(v10) = 1;
    return (unsigned int)v10;
  }
  if ( v15 != 31 )
  {
    if ( v15 == 32 )
    {
      v35 = **(_QWORD **)(a2 - 8);
      v87.m128i_i64[0] = 0;
      v87.m128i_i64[1] = 1;
      v79 = v35;
      v81 = *(__int64 **)(a2 + 40);
      v36 = (__int64 *)&v88;
      do
      {
        *v36 = -4096;
        v36 += 2;
      }
      while ( v36 != (__int64 *)&v95 );
      v37 = v81[6] & 0xFFFFFFFFFFFFFFF8LL;
      if ( (__int64 *)v37 != v81 + 6 )
      {
        if ( !v37 )
          BUG();
        v38 = v37 - 24;
        if ( (unsigned int)*(unsigned __int8 *)(v37 - 24) - 30 <= 0xA )
        {
          v39 = sub_B46E30(v38);
          if ( v39 )
          {
            v40 = 0;
            v41 = v39;
            while ( 1 )
            {
              v49 = sub_B46EC0(v38, v40);
              v84 = v49;
              if ( (v87.m128i_i8[8] & 1) != 0 )
              {
                v42 = 15;
                v43 = (__int64 *)&v88;
              }
              else
              {
                v50 = v89;
                v43 = v88;
                if ( !(_DWORD)v89 )
                {
                  v51 = v87.m128i_u32[2];
                  ++v87.m128i_i64[0];
                  v85 = 0;
                  v52 = ((unsigned __int32)v87.m128i_i32[2] >> 1) + 1;
                  goto LABEL_54;
                }
                v42 = v89 - 1;
              }
              v44 = v42 & (((unsigned int)v49 >> 9) ^ ((unsigned int)v49 >> 4));
              v45 = &v43[2 * v44];
              v46 = *v45;
              if ( v49 != *v45 )
                break;
LABEL_48:
              v47 = (int *)(v45 + 1);
              v48 = *((_DWORD *)v45 + 2) + 1;
LABEL_49:
              ++v40;
              *v47 = v48;
              if ( v41 == v40 )
              {
                v2 = a1;
                goto LABEL_61;
              }
            }
            v70 = 1;
            v71 = 0;
            while ( v46 != -4096 )
            {
              if ( !v71 && v46 == -8192 )
                v71 = v45;
              v44 = v42 & (v70 + v44);
              v45 = &v43[2 * v44];
              v46 = *v45;
              if ( v49 == *v45 )
                goto LABEL_48;
              ++v70;
            }
            v53 = 48;
            v50 = 16;
            if ( !v71 )
              v71 = v45;
            v51 = v87.m128i_u32[2];
            ++v87.m128i_i64[0];
            v85 = v71;
            v52 = ((unsigned __int32)v87.m128i_i32[2] >> 1) + 1;
            if ( (v87.m128i_i8[8] & 1) == 0 )
            {
              v50 = v89;
LABEL_54:
              v53 = 3 * v50;
            }
            if ( v53 <= 4 * v52 )
            {
              v50 *= 2;
            }
            else if ( v50 - v87.m128i_i32[3] - v52 > v50 >> 3 )
            {
LABEL_57:
              v87.m128i_i32[2] = (2 * (v51 >> 1) + 2) | v51 & 1;
              v54 = v85;
              if ( *v85 != -4096 )
                --v87.m128i_i32[3];
              *v85 = v49;
              v47 = (int *)(v54 + 1);
              *((_DWORD *)v54 + 2) = 0;
              v48 = 1;
              goto LABEL_49;
            }
            sub_2796C10((__int64)&v87, v50);
            sub_278FE70((__int64)&v87, &v84, &v85);
            v49 = v84;
            v51 = v87.m128i_u32[2];
            goto LABEL_57;
          }
        }
      }
LABEL_61:
      v55 = v87.m128i_i8[8];
      v56 = 0;
      LODWORD(v10) = 0;
      v57 = (*(_DWORD *)(a2 + 4) & 0x7FFFFFFu) >> 1;
      v58 = v57 - 1;
      if ( v57 == 1 )
      {
        v68 = v87.m128i_i8[8] & 1;
LABEL_73:
        if ( !v68 )
          sub_C7D6A0((__int64)v88, 16LL * (unsigned int)v89, 8);
        return (unsigned int)v10;
      }
      while ( 1 )
      {
        v65 = 32;
        if ( (_DWORD)v56 != -2 )
          v65 = 32LL * (unsigned int)(2 * v56 + 3);
        v66 = *(_QWORD *)(a2 - 8);
        ++v56;
        v67 = *(_QWORD *)(v66 + v65);
        v68 = v55 & 1;
        if ( (v55 & 1) != 0 )
          break;
        v59 = v88;
        if ( (_DWORD)v89 )
        {
          v60 = v89 - 1;
LABEL_64:
          v61 = v60 & (((unsigned int)v67 >> 9) ^ ((unsigned int)v67 >> 4));
          v62 = &v59[2 * v61];
          v63 = *v62;
          if ( *v62 == v67 )
          {
LABEL_65:
            if ( *((_DWORD *)v62 + 2) == 1 )
            {
              v86 = v67;
              v85 = v81;
              v64 = sub_2795CD0(v2, v79, *(_QWORD *)(v66 + 32LL * (unsigned int)(2 * v56)), (__int64 *)&v85, 1);
              v55 = v87.m128i_i8[8];
              LODWORD(v10) = v64 | v10;
              v68 = v87.m128i_i8[8] & 1;
            }
          }
          else
          {
            v69 = 1;
            while ( v63 != -4096 )
            {
              v61 = v60 & (v69 + v61);
              v78 = v69 + 1;
              v62 = &v59[2 * v61];
              v63 = *v62;
              if ( *v62 == v67 )
                goto LABEL_65;
              v69 = v78;
            }
          }
        }
        if ( v58 == v56 )
          goto LABEL_73;
      }
      v59 = (__int64 *)&v88;
      v60 = 15;
      goto LABEL_64;
    }
LABEL_33:
    if ( *(_BYTE *)(*(_QWORD *)(a2 + 8) + 8LL) == 7 )
      goto LABEL_29;
    v30 = *(_DWORD *)(a1 + 344);
    v31 = sub_2792F80(a1 + 136, a2);
    v32 = *(_BYTE *)a2 - 30;
    if ( v32 > 0x36u )
    {
      v33 = *(_QWORD *)(a2 + 40);
    }
    else
    {
      v10 = 0x400000400007FFuLL >> v32;
      v33 = *(_QWORD *)(a2 + 40);
      LODWORD(v10) = (v10 & 1) == 0;
      if ( !(_DWORD)v10 )
        goto LABEL_36;
    }
    if ( v31 >= v30 )
    {
      sub_27915B0(a1 + 352, v31, a2, v33);
      goto LABEL_29;
    }
    v83 = v31;
    v72 = sub_278BCD0(a1, v33, v31);
    v31 = v83;
    v10 = (__int64)v72;
    if ( v72 )
    {
      if ( v72 != (unsigned __int8 *)a2 )
      {
        sub_F57050((unsigned __int8 *)a2, v72);
        sub_BD84D0(a2, v10);
        v73 = *(_QWORD *)(a1 + 16);
        if ( v73 )
        {
          v74 = *(_QWORD *)(v10 + 8);
          if ( (unsigned int)*(unsigned __int8 *)(v74 + 8) - 17 <= 1 )
            v74 = **(_QWORD **)(v74 + 16);
          if ( *(_BYTE *)(v74 + 8) == 14 )
            sub_102B9D0(v73, v10);
        }
        sub_278A7A0(v2 + 136, (_BYTE *)a2);
        v77 = *(unsigned int *)(v2 + 656);
        if ( v77 + 1 > (unsigned __int64)*(unsigned int *)(v2 + 660) )
        {
          sub_C8D5F0(v2 + 648, (const void *)(v2 + 664), v77 + 1, 8u, v75, v76);
          v77 = *(unsigned int *)(v2 + 656);
        }
        *(_QWORD *)(*(_QWORD *)(v2 + 648) + 8 * v77) = a2;
        ++*(_DWORD *)(v2 + 656);
        goto LABEL_9;
      }
LABEL_29:
      LODWORD(v10) = 0;
      return (unsigned int)v10;
    }
    v33 = *(_QWORD *)(a2 + 40);
LABEL_36:
    sub_27915B0(a1 + 352, v31, a2, v33);
    return (unsigned int)v10;
  }
  if ( (*(_DWORD *)(a2 + 4) & 0x7FFFFFF) != 3 )
    goto LABEL_29;
  v17 = *(_BYTE **)(a2 - 96);
  if ( *v17 > 0x15u )
  {
    v18 = *(_QWORD *)(a2 - 32);
    LODWORD(v10) = 0;
    v82 = *(_QWORD *)(a2 - 64);
    if ( v18 != v82 )
    {
      v80 = *(_QWORD *)(a2 - 32);
      v19 = *(__int64 **)(a2 + 40);
      v20 = (__int64 *)sub_AA48A0(v18);
      v21 = sub_ACD6D0(v20);
      v85 = v19;
      v86 = v80;
      v22 = sub_2795CD0(a1, (__int64)v17, v21, (__int64 *)&v85, 1);
      v23 = (__int64 *)sub_AA48A0(v82);
      v24 = sub_ACD720(v23);
      v87.m128i_i64[0] = (__int64)v19;
      v87.m128i_i64[1] = v82;
      LODWORD(v10) = sub_2795CD0(a1, (__int64)v17, v24, v87.m128i_i64, 1) | v22;
    }
    return (unsigned int)v10;
  }
  return sub_279EB90(a1, a2);
}
