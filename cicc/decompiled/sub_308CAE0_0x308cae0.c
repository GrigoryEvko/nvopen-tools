// Function: sub_308CAE0
// Address: 0x308cae0
//
char __fastcall sub_308CAE0(_QWORD *a1, __int64 a2, int a3, __int64 a4)
{
  int v5; // eax
  __int64 v6; // rax
  __int64 v7; // rax
  _QWORD *v8; // rdi
  __int64 v9; // rax
  __int64 v10; // rsi
  __int64 v11; // r12
  __int64 v12; // r12
  __int64 v13; // rdi
  _QWORD *v14; // r13
  __int64 v15; // rdx
  __int64 v16; // r12
  __int64 v17; // r14
  __int32 v18; // eax
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rax
  char *v22; // rax
  __int64 v23; // rax
  unsigned __int64 v24; // r14
  int v25; // eax
  __int64 **v26; // rax
  __int64 v27; // rax
  unsigned __int64 v28; // rdx
  __int64 v29; // rdi
  unsigned __int64 v30; // rax
  int v31; // ecx
  unsigned __int16 v32; // si
  int v33; // eax
  __int64 v34; // rax
  unsigned __int64 v35; // rax
  __int64 v36; // rcx
  __int64 v37; // r8
  void *v38; // r9
  unsigned int v39; // esi
  __int64 v40; // rdi
  int v41; // r12d
  __int64 *v42; // rdx
  unsigned int v43; // ecx
  __int64 v44; // rax
  __int64 v45; // r9
  unsigned int *v46; // rdx
  __int64 v47; // rax
  char v48; // r9
  int v49; // r8d
  unsigned __int64 v50; // rax
  __int64 v51; // rdx
  unsigned __int64 v52; // rdx
  int v53; // ecx
  int v54; // eax
  int v55; // esi
  __int64 v56; // r8
  unsigned int v57; // eax
  __int64 v58; // rdi
  int v59; // r11d
  __int64 *v60; // r9
  int v61; // eax
  int v62; // eax
  __int64 v63; // rdi
  __int64 *v64; // r8
  unsigned int v65; // ebx
  int v66; // r9d
  __int64 v67; // rsi
  __int64 **v69; // [rsp+8h] [rbp-B8h]
  __int64 v70; // [rsp+10h] [rbp-B0h]
  unsigned int v74; // [rsp+2Ch] [rbp-94h]
  __int64 *v75; // [rsp+30h] [rbp-90h]
  __int64 v76; // [rsp+38h] [rbp-88h]
  __int128 v77; // [rsp+40h] [rbp-80h] BYREF
  __int64 v78; // [rsp+50h] [rbp-70h]
  __m128i v79; // [rsp+60h] [rbp-60h] BYREF
  __int64 v80; // [rsp+70h] [rbp-50h]
  __int64 v81; // [rsp+78h] [rbp-48h]
  __int64 v82; // [rsp+80h] [rbp-40h]

  v75 = *(__int64 **)(*(_QWORD *)(a2 + 24) + 32LL);
  if ( (unsigned int)*(unsigned __int16 *)(a2 + 68) - 1 <= 1 && (*(_BYTE *)(*(_QWORD *)(a2 + 32) + 64LL) & 8) != 0
    || ((v5 = *(_DWORD *)(a2 + 44), (v5 & 4) != 0) || (v5 & 8) == 0
      ? (v6 = (*(_QWORD *)(*(_QWORD *)(a2 + 16) + 24LL) >> 19) & 1LL)
      : (LOBYTE(v6) = sub_2E88A90(a2, 0x80000, 1)),
        (_BYTE)v6) )
  {
    v8 = (_QWORD *)sub_B2BE50(*v75);
    v9 = *(unsigned __int16 *)(a2 + 68);
    switch ( (__int16)v9 )
    {
      case 2659:
        v70 = 2;
        v69 = (__int64 **)sub_BCB160(v8);
        v9 = *(unsigned __int16 *)(a2 + 68);
        v74 = 2;
        break;
      case 2660:
        v70 = 4;
        v69 = (__int64 **)sub_BCB160(v8);
        v9 = *(unsigned __int16 *)(a2 + 68);
        v74 = 4;
        break;
      case 2662:
        v70 = 2;
        v69 = (__int64 **)sub_BCB170(v8);
        v9 = *(unsigned __int16 *)(a2 + 68);
        v74 = 2;
        break;
      case 2663:
        v70 = 4;
        v74 = 4;
        v69 = (__int64 **)sub_BCB170(v8);
        v9 = *(unsigned __int16 *)(a2 + 68);
        break;
      case 2664:
        v70 = 2;
        v69 = (__int64 **)sub_BCB2C0(v8);
        v9 = *(unsigned __int16 *)(a2 + 68);
        v74 = 2;
        break;
      case 2665:
        v70 = 4;
        v69 = (__int64 **)sub_BCB2C0(v8);
        v9 = *(unsigned __int16 *)(a2 + 68);
        v74 = 4;
        break;
      case 2666:
        v70 = 2;
        v69 = (__int64 **)sub_BCB2D0(v8);
        v9 = *(unsigned __int16 *)(a2 + 68);
        v74 = 2;
        break;
      case 2667:
        v70 = 4;
        v69 = (__int64 **)sub_BCB2D0(v8);
        v9 = *(unsigned __int16 *)(a2 + 68);
        v74 = 4;
        break;
      case 2669:
        v70 = 2;
        v69 = (__int64 **)sub_BCB2E0(v8);
        v9 = *(unsigned __int16 *)(a2 + 68);
        v74 = 2;
        break;
      case 2670:
        v70 = 4;
        v69 = (__int64 **)sub_BCB2E0(v8);
        v9 = *(unsigned __int16 *)(a2 + 68);
        v74 = 4;
        break;
      case 2671:
        v70 = 2;
        v69 = (__int64 **)sub_BCB2B0(v8);
        v9 = *(unsigned __int16 *)(a2 + 68);
        v74 = 2;
        break;
      case 2672:
        v70 = 4;
        v69 = (__int64 **)sub_BCB2B0(v8);
        v9 = *(unsigned __int16 *)(a2 + 68);
        v74 = 4;
        break;
      case 2673:
        v70 = 1;
        v69 = (__int64 **)sub_BCB160(v8);
        v9 = *(unsigned __int16 *)(a2 + 68);
        v74 = 1;
        break;
      case 2674:
        v70 = 1;
        v69 = (__int64 **)sub_BCB170(v8);
        v9 = *(unsigned __int16 *)(a2 + 68);
        v74 = 1;
        break;
      case 2675:
        v70 = 1;
        v69 = (__int64 **)sub_BCB2C0(v8);
        v9 = *(unsigned __int16 *)(a2 + 68);
        v74 = 1;
        break;
      case 2676:
        v70 = 1;
        v69 = (__int64 **)sub_BCB2D0(v8);
        v9 = *(unsigned __int16 *)(a2 + 68);
        v74 = 1;
        break;
      case 2677:
        v70 = 1;
        v69 = (__int64 **)sub_BCB2E0(v8);
        v9 = *(unsigned __int16 *)(a2 + 68);
        v74 = 1;
        break;
      case 2678:
        v70 = 1;
        v69 = (__int64 **)sub_BCB2B0(v8);
        v9 = *(unsigned __int16 *)(a2 + 68);
        v74 = 1;
        break;
      default:
        break;
    }
    v10 = *(_QWORD *)(a2 + 56);
    v11 = *(_QWORD *)(a1[58] + 8LL);
    *(_QWORD *)&v77 = v10;
    v12 = v11 - 40 * v9;
    if ( v10 )
    {
      sub_B96E90((__int64)&v77, v10, 1);
      v79.m128i_i64[0] = v77;
      if ( (_QWORD)v77 )
      {
        sub_B976B0((__int64)&v77, (unsigned __int8 *)v77, (__int64)&v79);
        *(_QWORD *)&v77 = 0;
      }
    }
    else
    {
      v79.m128i_i64[0] = 0;
    }
    v13 = *(_QWORD *)(a2 + 24);
    v79.m128i_i64[1] = 0;
    v80 = 0;
    v14 = sub_301D240(v13, a2, (__int64)&v79, v12);
    v16 = v15;
    if ( v79.m128i_i64[0] )
      sub_B91220((__int64)&v79, v79.m128i_i64[0]);
    if ( (_QWORD)v77 )
      sub_B91220((__int64)&v77, v77);
    v17 = 0;
    v76 = 40LL * v74;
    do
    {
      v18 = *(_DWORD *)(*(_QWORD *)(a2 + 32) + v17 + 8);
      v79.m128i_i64[0] = 0x10000000;
      v17 += 40;
      v80 = 0;
      v79.m128i_i32[2] = v18;
      v81 = 0;
      v82 = 0;
      sub_2E8EAD0(v16, (__int64)v14, &v79);
    }
    while ( v76 != v17 );
    v19 = *(_QWORD *)(*(_QWORD *)(a2 + 32) + v76 + 24);
    v79.m128i_i64[0] = 1;
    v80 = 0;
    v81 = v19;
    sub_2E8EAD0(v16, (__int64)v14, &v79);
    v79.m128i_i64[0] = 1;
    v80 = 0;
    v81 = 0;
    sub_2E8EAD0(v16, (__int64)v14, &v79);
    v79.m128i_i64[0] = 1;
    v80 = 0;
    v81 = 101;
    sub_2E8EAD0(v16, (__int64)v14, &v79);
    v79.m128i_i64[0] = 1;
    v80 = 0;
    v81 = v70;
    sub_2E8EAD0(v16, (__int64)v14, &v79);
    v20 = *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL * (v74 + 4) + 24);
    v79.m128i_i64[0] = 1;
    v80 = 0;
    v81 = v20;
    sub_2E8EAD0(v16, (__int64)v14, &v79);
    v21 = *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL * (v74 + 5) + 24);
    v79.m128i_i64[0] = 1;
    v80 = 0;
    v81 = v21;
    sub_2E8EAD0(v16, (__int64)v14, &v79);
    v22 = sub_3045CE0(
            a1[59],
            v75,
            858993459 * (unsigned int)((__int64)(*(_QWORD *)(a1[55] + 16LL) - *(_QWORD *)(a1[55] + 8LL)) >> 3) + a3);
    v79.m128i_i8[0] = 9;
    v81 = (__int64)v22;
    v79.m128i_i32[0] &= 0xFFF000FF;
    v80 = 0;
    v79.m128i_i32[2] = 0;
    LODWORD(v82) = 0;
    sub_2E8EAD0(v16, (__int64)v14, &v79);
    if ( v74 + 8 == (*(_DWORD *)(a2 + 40) & 0xFFFFFF) )
    {
      v47 = *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL * (v74 + 7) + 24);
      v79.m128i_i64[0] = 1;
      v80 = 0;
      v81 = v47;
    }
    else
    {
      v79.m128i_i64[0] = 1;
      v80 = 0;
      v81 = 0;
    }
    sub_2E8EAD0(v16, (__int64)v14, &v79);
    v23 = *(_QWORD *)(a2 + 48);
    v24 = v23 & 0xFFFFFFFFFFFFFFF8LL;
    if ( (v23 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
LABEL_106:
      BUG();
    v25 = v23 & 7;
    if ( v25 )
    {
      if ( v25 != 3 )
        goto LABEL_106;
      v24 = *(_QWORD *)(v24 + 16);
    }
    else
    {
      *(_QWORD *)(a2 + 48) = v24;
    }
    v26 = (__int64 **)sub_BCE760(v69, 0);
    v27 = sub_AC9EC0(v26);
    v79 = 0u;
    LODWORD(v28) = -1;
    v29 = v27;
    v80 = 0;
    v81 = 0;
    v30 = *(_QWORD *)(v24 + 24);
    v31 = *(unsigned __int8 *)(v24 + 34);
    if ( (v30 & 0xFFFFFFFFFFFFFFF9LL) != 0 )
    {
      v48 = *(_BYTE *)(v24 + 24) & 2;
      if ( (*(_BYTE *)(v24 + 24) & 6) == 2 || (*(_BYTE *)(v24 + 24) & 1) != 0 )
      {
        v52 = HIWORD(v30);
        if ( !v48 )
          v52 = HIDWORD(v30);
        v28 = (v52 + 7) >> 3;
      }
      else
      {
        v49 = (unsigned __int16)((unsigned int)v30 >> 8);
        v50 = HIWORD(v30);
        v51 = HIDWORD(*(_QWORD *)(v24 + 24));
        if ( v48 )
          LODWORD(v51) = v50;
        v28 = ((unsigned __int64)(unsigned int)(v49 * v51) + 7) >> 3;
      }
    }
    v32 = *(_WORD *)(v24 + 32);
    *((_QWORD *)&v77 + 1) = 0;
    BYTE4(v78) = 0;
    *(_QWORD *)&v77 = v29 & 0xFFFFFFFFFFFFFFFBLL;
    v33 = 0;
    if ( v29 )
    {
      v34 = *(_QWORD *)(v29 + 8);
      if ( (unsigned int)*(unsigned __int8 *)(v34 + 8) - 17 <= 1 )
        v34 = **(_QWORD **)(v34 + 16);
      v33 = *(_DWORD *)(v34 + 8) >> 8;
    }
    LODWORD(v78) = v33;
    v35 = sub_2E7BD70(v75, v32, v28, v31, (int)&v79, 0, v77, v78, 1u, 0, 0);
    sub_2E86C70(v16, (__int64)v14, v35, v36, v37, v38);
    v39 = *(_DWORD *)(a4 + 24);
    if ( v39 )
    {
      v40 = *(_QWORD *)(a4 + 8);
      v41 = 1;
      v42 = 0;
      v43 = (v39 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v44 = v40 + 16LL * v43;
      v45 = *(_QWORD *)v44;
      if ( *(_QWORD *)v44 == a2 )
      {
LABEL_35:
        v46 = (unsigned int *)(v44 + 8);
        v74 += *(_DWORD *)(v44 + 8);
LABEL_36:
        LOBYTE(v7) = v74;
        *v46 = v74;
        return v7;
      }
      while ( v45 != -4096 )
      {
        if ( !v42 && v45 == -8192 )
          v42 = (__int64 *)v44;
        v43 = (v39 - 1) & (v41 + v43);
        v44 = v40 + 16LL * v43;
        v45 = *(_QWORD *)v44;
        if ( *(_QWORD *)v44 == a2 )
          goto LABEL_35;
        ++v41;
      }
      if ( !v42 )
        v42 = (__int64 *)v44;
      ++*(_QWORD *)a4;
      v53 = *(_DWORD *)(a4 + 16) + 1;
      if ( 4 * v53 < 3 * v39 )
      {
        if ( v39 - *(_DWORD *)(a4 + 20) - v53 > v39 >> 3 )
        {
LABEL_78:
          *(_DWORD *)(a4 + 16) = v53;
          if ( *v42 != -4096 )
            --*(_DWORD *)(a4 + 20);
          *v42 = a2;
          v46 = (unsigned int *)(v42 + 1);
          *v46 = 0;
          goto LABEL_36;
        }
        sub_2E261E0(a4, v39);
        v61 = *(_DWORD *)(a4 + 24);
        if ( v61 )
        {
          v62 = v61 - 1;
          v63 = *(_QWORD *)(a4 + 8);
          v64 = 0;
          v65 = v62 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
          v66 = 1;
          v53 = *(_DWORD *)(a4 + 16) + 1;
          v42 = (__int64 *)(v63 + 16LL * v65);
          v67 = *v42;
          if ( *v42 != a2 )
          {
            while ( v67 != -4096 )
            {
              if ( v67 == -8192 && !v64 )
                v64 = v42;
              v65 = v62 & (v66 + v65);
              v42 = (__int64 *)(v63 + 16LL * v65);
              v67 = *v42;
              if ( *v42 == a2 )
                goto LABEL_78;
              ++v66;
            }
            if ( v64 )
              v42 = v64;
          }
          goto LABEL_78;
        }
        goto LABEL_105;
      }
    }
    else
    {
      ++*(_QWORD *)a4;
    }
    sub_2E261E0(a4, 2 * v39);
    v54 = *(_DWORD *)(a4 + 24);
    if ( v54 )
    {
      v55 = v54 - 1;
      v56 = *(_QWORD *)(a4 + 8);
      v57 = (v54 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v53 = *(_DWORD *)(a4 + 16) + 1;
      v42 = (__int64 *)(v56 + 16LL * v57);
      v58 = *v42;
      if ( *v42 != a2 )
      {
        v59 = 1;
        v60 = 0;
        while ( v58 != -4096 )
        {
          if ( !v60 && v58 == -8192 )
            v60 = v42;
          v57 = v55 & (v59 + v57);
          v42 = (__int64 *)(v56 + 16LL * v57);
          v58 = *v42;
          if ( *v42 == a2 )
            goto LABEL_78;
          ++v59;
        }
        if ( v60 )
          v42 = v60;
      }
      goto LABEL_78;
    }
LABEL_105:
    ++*(_DWORD *)(a4 + 16);
    goto LABEL_106;
  }
  if ( (unsigned int)*(unsigned __int16 *)(a2 + 68) - 1 > 1
    || (v7 = *(_QWORD *)(a2 + 32), (*(_BYTE *)(v7 + 64) & 0x10) == 0) )
  {
    LODWORD(v7) = *(_DWORD *)(a2 + 44);
    if ( (v7 & 4) == 0 && (v7 & 8) != 0 )
      LOBYTE(v7) = sub_2E88A90(a2, 0x100000, 1);
  }
  return v7;
}
