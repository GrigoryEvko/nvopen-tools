// Function: sub_275BFC0
// Address: 0x275bfc0
//
_BOOL8 __fastcall sub_275BFC0(__int64 a1)
{
  __int64 v2; // rbx
  __int64 v3; // rax
  __int64 v4; // r15
  __int64 v5; // r13
  _QWORD *v6; // rax
  _QWORD *v7; // rdx
  unsigned __int8 *v9; // r12
  char v10; // al
  unsigned __int8 *v11; // rax
  unsigned __int64 v12; // rcx
  __int64 v13; // r9
  unsigned __int8 *v14; // r8
  int v15; // esi
  __int64 v16; // rdi
  unsigned int v17; // edx
  unsigned __int64 *v18; // rax
  char v19; // al
  __int64 v20; // rdi
  int v21; // r13d
  unsigned int v22; // eax
  __int64 v23; // rdx
  __int64 v24; // r12
  char v25; // bl
  unsigned __int8 *v26; // r15
  unsigned __int64 v27; // rcx
  __int64 v28; // r8
  __int64 v29; // r9
  char v30; // al
  __int64 v31; // rcx
  __int64 v32; // rax
  __m128i v33; // xmm0
  _QWORD *v34; // rdi
  __m128i v35; // xmm1
  __m128i v36; // xmm2
  unsigned __int8 **v37; // r11
  int v38; // eax
  int v39; // eax
  char v40; // al
  __int64 *v41; // r11
  unsigned __int8 *v42; // r8
  char v43; // al
  char v44; // al
  int v45; // eax
  int v46; // r9d
  __int64 v47; // r10
  unsigned int v48; // edx
  unsigned __int8 *v49; // rdi
  int v50; // esi
  unsigned __int8 **v51; // rcx
  int v52; // eax
  int v53; // r9d
  __int64 v54; // r10
  int v55; // esi
  unsigned int v56; // edx
  unsigned __int8 *v57; // rdi
  __int64 v58; // [rsp+8h] [rbp-188h]
  unsigned __int8 *v59; // [rsp+10h] [rbp-180h]
  __int64 v60; // [rsp+18h] [rbp-178h]
  unsigned int v61; // [rsp+18h] [rbp-178h]
  __int64 v62; // [rsp+18h] [rbp-178h]
  int v63; // [rsp+20h] [rbp-170h]
  unsigned __int8 *v64; // [rsp+20h] [rbp-170h]
  __int64 *v65; // [rsp+20h] [rbp-170h]
  __int64 *v66; // [rsp+20h] [rbp-170h]
  unsigned __int8 *v67; // [rsp+20h] [rbp-170h]
  unsigned int v68; // [rsp+28h] [rbp-168h]
  __int64 *v69; // [rsp+28h] [rbp-168h]
  __int64 v70; // [rsp+28h] [rbp-168h]
  __int64 v71; // [rsp+28h] [rbp-168h]
  unsigned __int8 *v72; // [rsp+28h] [rbp-168h]
  unsigned __int8 *v73; // [rsp+30h] [rbp-160h]
  bool v74; // [rsp+45h] [rbp-14Bh]
  unsigned __int8 v75; // [rsp+46h] [rbp-14Ah]
  bool v76; // [rsp+47h] [rbp-149h]
  _BYTE *v77; // [rsp+50h] [rbp-140h] BYREF
  __int64 v78; // [rsp+58h] [rbp-138h]
  _BYTE v79[32]; // [rsp+60h] [rbp-130h] BYREF
  __m128i v80; // [rsp+80h] [rbp-110h] BYREF
  __m128i v81; // [rsp+90h] [rbp-100h] BYREF
  __m128i v82; // [rsp+A0h] [rbp-F0h] BYREF
  char v83; // [rsp+B0h] [rbp-E0h]
  __m128i v84[3]; // [rsp+C0h] [rbp-D0h] BYREF
  char v85; // [rsp+F0h] [rbp-A0h]
  __int64 v86; // [rsp+100h] [rbp-90h] BYREF
  char *v87; // [rsp+108h] [rbp-88h]
  __int64 v88; // [rsp+110h] [rbp-80h]
  int v89; // [rsp+118h] [rbp-78h]
  char v90; // [rsp+11Ch] [rbp-74h]
  char v91; // [rsp+120h] [rbp-70h] BYREF

  v74 = 0;
  do
  {
    v2 = *(_QWORD *)(a1 + 840);
    v3 = *(unsigned int *)(a1 + 848);
    *(_BYTE *)(a1 + 1737) = 0;
    v4 = v2 + 8 * v3;
    if ( v2 == v4 )
      break;
    do
    {
      while ( 1 )
      {
        while ( 1 )
        {
          v5 = *(_QWORD *)(v4 - 8);
          if ( *(_BYTE *)(a1 + 1396) )
          {
            v6 = *(_QWORD **)(a1 + 1376);
            v7 = &v6[*(unsigned int *)(a1 + 1388)];
            if ( v6 != v7 )
            {
              while ( v5 != *v6 )
              {
                if ( v7 == ++v6 )
                  goto LABEL_12;
              }
              goto LABEL_8;
            }
          }
          else if ( sub_C8CA60(a1 + 1368, *(_QWORD *)(v4 - 8)) )
          {
            goto LABEL_8;
          }
LABEL_12:
          v9 = *(unsigned __int8 **)(v5 + 72);
          if ( !(unsigned __int8)sub_B46490((__int64)v9) )
            goto LABEL_8;
          if ( (unsigned __int8)(*v9 - 34) <= 0x33u
            && (v31 = 0x8000000000041LL, _bittest64(&v31, (unsigned int)*v9 - 34)) )
          {
            sub_D67230(&v80, v9, *(__int64 **)(a1 + 808));
            v10 = v83;
          }
          else
          {
            sub_D66840(&v80, v9);
            v10 = v83;
          }
          if ( !v10 )
            goto LABEL_8;
          v76 = sub_27543D0(v9);
          if ( !v76 )
            goto LABEL_8;
          v73 = (unsigned __int8 *)v80.m128i_i64[0];
          v11 = sub_98ACB0((unsigned __int8 *)v80.m128i_i64[0], 6u);
          v14 = v11;
          if ( *v11 != 60 )
            break;
LABEL_22:
          v20 = v5;
          v86 = 0;
          v21 = 0;
          v77 = v79;
          v78 = 0x400000000LL;
          v87 = &v91;
          v88 = 8;
          v89 = 0;
          v90 = 1;
          sub_2754040(v20, (__int64)&v77, &v86, v12, (__int64)v14, v13);
          v22 = v78;
          v23 = 0;
          if ( !(_DWORD)v78 )
          {
LABEL_40:
            if ( !v90 )
              goto LABEL_35;
            goto LABEL_41;
          }
          v60 = v2;
          v59 = v9;
          v58 = v4;
          do
          {
            if ( v22 >= dword_4FFA9A8 )
            {
LABEL_39:
              v76 = 0;
              v2 = v60;
              v9 = v59;
              v4 = v58;
              goto LABEL_40;
            }
            v24 = *(_QWORD *)&v77[8 * v23];
            v25 = *(_BYTE *)v24;
            if ( *(_BYTE *)v24 != 28 )
            {
              v26 = *(unsigned __int8 **)(v24 + 72);
              if ( !sub_2753FC0((__int64)v26) )
              {
                v27 = *v26;
                if ( (_BYTE)v27 == 62 )
                {
                  v30 = byte_3F70480[8 * ((*((_WORD *)v26 + 1) >> 7) & 7) + 2];
LABEL_29:
                  if ( v30 )
                    goto LABEL_39;
                  goto LABEL_30;
                }
                v75 = *v26;
                if ( (unsigned __int8)sub_B46420((__int64)v26) )
                {
                  if ( (unsigned __int8)(v75 - 34) > 0x33u
                    || (v32 = 0x8000000000041LL, !_bittest64(&v32, (unsigned int)v75 - 34))
                    || !sub_B49EA0((__int64)v26) )
                  {
                    v33 = _mm_loadu_si128(&v80);
                    v34 = *(_QWORD **)(a1 + 104);
                    v35 = _mm_loadu_si128(&v81);
                    v36 = _mm_loadu_si128(&v82);
                    v85 = 1;
                    v84[0] = v33;
                    v84[1] = v35;
                    v84[2] = v36;
                    v30 = sub_CF63E0(v34, v26, v84, a1 + 112) & 1;
                    goto LABEL_29;
                  }
LABEL_30:
                  v25 = *(_BYTE *)v24;
                }
              }
              if ( v25 != 27 )
                goto LABEL_33;
              goto LABEL_32;
            }
            if ( !sub_2753D10(a1, v73) )
              goto LABEL_39;
LABEL_32:
            sub_2754040(v24, (__int64)&v77, &v86, v27, v28, v29);
LABEL_33:
            v22 = v78;
            v23 = (unsigned int)(v21 + 1);
            v21 = v23;
          }
          while ( (unsigned int)v78 > (unsigned int)v23 );
          v2 = v60;
          v9 = v59;
          v4 = v58;
          if ( v90 )
            goto LABEL_41;
LABEL_35:
          _libc_free((unsigned __int64)v87);
LABEL_41:
          if ( v77 != v79 )
            _libc_free((unsigned __int64)v77);
          if ( !v76 )
            goto LABEL_8;
          v4 -= 8;
          sub_2754990(a1, (__int64)v9, 0);
          v74 = v76;
          if ( v2 == v4 )
            goto LABEL_9;
        }
        v15 = *(_DWORD *)(a1 + 1488);
        if ( v15 )
        {
          v13 = (unsigned int)(v15 - 1);
          v16 = *(_QWORD *)(a1 + 1472);
          v68 = ((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4);
          v17 = v13 & v68;
          v18 = (unsigned __int64 *)(v16 + 16LL * ((unsigned int)v13 & v68));
          v12 = *v18;
          if ( v14 == (unsigned __int8 *)*v18 )
          {
LABEL_20:
            v19 = *((_BYTE *)v18 + 8);
            goto LABEL_21;
          }
          v63 = 1;
          v37 = 0;
          v61 = *(_DWORD *)(a1 + 1488);
          while ( v12 != -4096 )
          {
            if ( !v37 && v12 == -8192 )
              v37 = (unsigned __int8 **)v18;
            v17 = v13 & (v63 + v17);
            v18 = (unsigned __int64 *)(v16 + 16LL * v17);
            v12 = *v18;
            if ( v14 == (unsigned __int8 *)*v18 )
              goto LABEL_20;
            ++v63;
          }
          if ( !v37 )
            v37 = (unsigned __int8 **)v18;
          v38 = *(_DWORD *)(a1 + 1480);
          ++*(_QWORD *)(a1 + 1464);
          v39 = v38 + 1;
          if ( 4 * v39 < 3 * v61 )
          {
            if ( v61 - *(_DWORD *)(a1 + 1484) - v39 <= v61 >> 3 )
            {
              v67 = v14;
              sub_275AA00(a1 + 1464, v61);
              v52 = *(_DWORD *)(a1 + 1488);
              if ( !v52 )
              {
LABEL_96:
                ++*(_DWORD *)(a1 + 1480);
                BUG();
              }
              v53 = v52 - 1;
              v51 = 0;
              v54 = *(_QWORD *)(a1 + 1472);
              v14 = v67;
              v55 = 1;
              v56 = v53 & v68;
              v39 = *(_DWORD *)(a1 + 1480) + 1;
              v37 = (unsigned __int8 **)(v54 + 16LL * (v53 & v68));
              v57 = *v37;
              if ( v67 != *v37 )
              {
                while ( v57 != (unsigned __int8 *)-4096LL )
                {
                  if ( v57 == (unsigned __int8 *)-8192LL && !v51 )
                    v51 = v37;
                  v56 = v53 & (v55 + v56);
                  v37 = (unsigned __int8 **)(v54 + 16LL * v56);
                  v57 = *v37;
                  if ( v67 == *v37 )
                    goto LABEL_57;
                  ++v55;
                }
LABEL_75:
                if ( v51 )
                  v37 = v51;
                goto LABEL_57;
              }
            }
            goto LABEL_57;
          }
        }
        else
        {
          ++*(_QWORD *)(a1 + 1464);
        }
        v72 = v14;
        sub_275AA00(a1 + 1464, 2 * v15);
        v45 = *(_DWORD *)(a1 + 1488);
        if ( !v45 )
          goto LABEL_96;
        v14 = v72;
        v46 = v45 - 1;
        v47 = *(_QWORD *)(a1 + 1472);
        v48 = (v45 - 1) & (((unsigned int)v72 >> 9) ^ ((unsigned int)v72 >> 4));
        v39 = *(_DWORD *)(a1 + 1480) + 1;
        v37 = (unsigned __int8 **)(v47 + 16LL * v48);
        v49 = *v37;
        if ( v72 != *v37 )
        {
          v50 = 1;
          v51 = 0;
          while ( v49 != (unsigned __int8 *)-4096LL )
          {
            if ( !v51 && v49 == (unsigned __int8 *)-8192LL )
              v51 = v37;
            v48 = v46 & (v50 + v48);
            v37 = (unsigned __int8 **)(v47 + 16LL * v48);
            v49 = *v37;
            if ( v72 == *v37 )
              goto LABEL_57;
            ++v50;
          }
          goto LABEL_75;
        }
LABEL_57:
        *(_DWORD *)(a1 + 1480) = v39;
        if ( *v37 != (unsigned __int8 *)-4096LL )
          --*(_DWORD *)(a1 + 1484);
        *v37 = v14;
        *((_BYTE *)v37 + 8) = 0;
        v69 = (__int64 *)v37;
        v64 = v14;
        v40 = sub_CF7590(v14, &v77);
        v41 = v69;
        if ( !v40 )
          break;
        v42 = v64;
        if ( (_BYTE)v77 )
        {
          v65 = v69;
          v84[0].m128i_i64[0] = (__int64)v42;
          v70 = (__int64)v42;
          v84[0].m128i_i8[8] = 1;
          sub_275ABE0((__int64)&v86, a1 + 1432, v84[0].m128i_i64, &v84[0].m128i_i8[8]);
          v42 = (unsigned __int8 *)v70;
          v41 = v65;
          if ( v91 )
          {
            v62 = v88;
            v43 = sub_D13FA0(v70, 0, 0);
            v41 = v65;
            v42 = (unsigned __int8 *)v70;
            *(_BYTE *)(v62 + 8) = v43;
          }
          else
          {
            v43 = *(_BYTE *)(v88 + 8);
          }
          if ( v43 )
            break;
        }
        v66 = v41;
        v71 = (__int64)v42;
        v44 = sub_CF6FD0(v42);
        v14 = (unsigned __int8 *)v71;
        if ( v44 )
        {
          v19 = sub_D13FA0(v71, 1, 0) ^ 1;
          *((_BYTE *)v66 + 8) = v19;
        }
        else
        {
          v19 = *((_BYTE *)v66 + 8);
        }
LABEL_21:
        if ( v19 )
          goto LABEL_22;
LABEL_8:
        v4 -= 8;
        if ( v2 == v4 )
          goto LABEL_9;
      }
      v4 -= 8;
      *((_BYTE *)v41 + 8) = 0;
    }
    while ( v2 != v4 );
LABEL_9:
    ;
  }
  while ( *(_BYTE *)(a1 + 1737) );
  return v74;
}
