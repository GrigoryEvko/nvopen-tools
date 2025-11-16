// Function: sub_3761A10
// Address: 0x3761a10
//
__int64 __fastcall sub_3761A10(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // rbx
  const __m128i *v3; // rax
  __int64 v4; // r12
  __int64 v5; // rax
  __int64 v6; // r8
  __int64 v7; // r9
  __m128i v8; // xmm0
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // r12
  __int64 i; // r14
  __int64 v15; // rax
  unsigned __int64 v16; // rcx
  __int64 v17; // rsi
  unsigned __int64 v18; // r14
  int v19; // edx
  __int64 v20; // rdx
  int v21; // eax
  unsigned int v22; // r12d
  __int64 v23; // r12
  __int64 v24; // rdx
  __int64 v25; // r12
  __int64 v26; // r13
  int v27; // eax
  __int64 v28; // rax
  __int64 v29; // rbx
  __int64 v30; // r12
  __int64 v31; // r13
  __int64 v33; // r12
  __int64 v34; // rbx
  unsigned int *v35; // rdx
  __int64 v36; // rdx
  char v37; // al
  __int64 v38; // rdx
  __int64 *v39; // r13
  unsigned int v40; // ebx
  int v41; // r12d
  __int64 v42; // rdx
  __int64 v43; // r8
  __int64 v44; // [rsp+10h] [rbp-170h]
  unsigned __int64 v45; // [rsp+18h] [rbp-168h]
  unsigned __int64 v46; // [rsp+20h] [rbp-160h]
  unsigned __int8 v47; // [rsp+2Fh] [rbp-151h]
  __m128i v48; // [rsp+30h] [rbp-150h] BYREF
  __int64 v49; // [rsp+40h] [rbp-140h]
  __int64 v50; // [rsp+48h] [rbp-138h]
  __int64 v51; // [rsp+50h] [rbp-130h]
  __int64 v52; // [rsp+58h] [rbp-128h]
  __int64 v53; // [rsp+60h] [rbp-120h]
  __int64 v54; // [rsp+68h] [rbp-118h]
  __m128i v55; // [rsp+70h] [rbp-110h]
  __int64 v56; // [rsp+80h] [rbp-100h] BYREF
  int v57; // [rsp+88h] [rbp-F8h]
  __int64 v58; // [rsp+90h] [rbp-F0h]
  _BYTE v59[32]; // [rsp+A0h] [rbp-E0h] BYREF
  _QWORD v60[4]; // [rsp+C0h] [rbp-C0h] BYREF
  __int64 v61; // [rsp+E0h] [rbp-A0h]
  __int64 *v62; // [rsp+E8h] [rbp-98h]
  __int64 v63; // [rsp+F0h] [rbp-90h]
  __int64 v64; // [rsp+F8h] [rbp-88h]
  __int64 v65; // [rsp+100h] [rbp-80h]
  int v66; // [rsp+108h] [rbp-78h]
  __int64 v67; // [rsp+110h] [rbp-70h]
  __int64 v68; // [rsp+118h] [rbp-68h]
  __int64 v69; // [rsp+120h] [rbp-60h] BYREF
  __int64 v70; // [rsp+128h] [rbp-58h]
  _QWORD *v71; // [rsp+130h] [rbp-50h]
  __int64 v72; // [rsp+138h] [rbp-48h]
  __int64 v73; // [rsp+140h] [rbp-40h] BYREF

  v3 = *(const __m128i **)(a1 + 8);
  v4 = v3[24].m128i_i64[0];
  v48 = _mm_loadu_si128(v3 + 24);
  v5 = sub_33ECD10(1u);
  v8 = _mm_load_si128(&v48);
  v73 = 0;
  v63 = v5;
  v65 = 0x100000000LL;
  v68 = 0xFFFFFFFFLL;
  v71 = v60;
  v55 = v8;
  v64 = 0;
  v66 = 0;
  v67 = 0;
  v72 = 0;
  LODWORD(v70) = v8.m128i_i32[2];
  v69 = v8.m128i_i64[0];
  v9 = *(_QWORD *)(v4 + 56);
  memset(v60, 0, 24);
  v60[3] = 328;
  v61 = 4294901760LL;
  v73 = v9;
  if ( v9 )
    *(_QWORD *)(v9 + 24) = &v73;
  v72 = v4 + 56;
  *(_QWORD *)(v4 + 56) = &v69;
  v62 = &v69;
  v10 = *(_QWORD *)(a1 + 8);
  v54 = 0;
  v53 = 0;
  *(_QWORD *)(v10 + 384) = 0;
  v11 = (unsigned int)v54;
  LODWORD(v65) = 1;
  *(_DWORD *)(v10 + 392) = v54;
  v12 = *(_QWORD *)(a1 + 8);
  HIDWORD(v61) = -2;
  v13 = *(_QWORD *)(v12 + 408);
  for ( i = v12 + 400; i != v13; v13 = *(_QWORD *)(v13 + 8) )
  {
    while ( 1 )
    {
      if ( !v13 )
        BUG();
      v11 = *(unsigned int *)(v13 + 56);
      if ( !(_DWORD)v11 )
        break;
      *(_DWORD *)(v13 + 28) = -2;
      v13 = *(_QWORD *)(v13 + 8);
      if ( i == v13 )
        goto LABEL_11;
    }
    *(_DWORD *)(v13 + 28) = 0;
    v15 = *(unsigned int *)(a1 + 1616);
    if ( v15 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 1620) )
    {
      a2 = a1 + 1624;
      sub_C8D5F0(a1 + 1608, (const void *)(a1 + 1624), v15 + 1, 8u, v6, v7);
      v15 = *(unsigned int *)(a1 + 1616);
    }
    v11 = *(_QWORD *)(a1 + 1608);
    *(_QWORD *)(v11 + 8 * v15) = v13 - 8;
    ++*(_DWORD *)(a1 + 1616);
  }
LABEL_11:
  v16 = *(unsigned int *)(a1 + 1616);
  v47 = 0;
  v48.m128i_i64[0] = (__int64)&v56;
  if ( (_DWORD)v16 )
  {
    while ( 1 )
    {
      if ( (_BYTE)qword_5050F68 )
        sub_37594D0(a1, a2, v11, v16, v6, v7);
      v16 = *(unsigned int *)(a1 + 1616);
      v17 = v48.m128i_i64[0];
      v18 = *(_QWORD *)(*(_QWORD *)(a1 + 1608) + 8 * v16 - 8);
      --*(_DWORD *)(a1 + 1616);
      v19 = *(_DWORD *)(v18 + 28) & 0xFE0;
      v56 = *(_QWORD *)(a1 + 8);
      v57 = v19;
      v20 = *(_QWORD *)(v56 + 1024);
      *(_QWORD *)(v56 + 1024) = v17;
      v21 = *(_DWORD *)(v18 + 24);
      v58 = v20;
      if ( v21 != 9 && v21 != 35 )
      {
        v22 = *(_DWORD *)(v18 + 68);
        if ( v22 )
        {
          v16 = v46;
          v44 = v22;
          v23 = 0;
          while ( 2 )
          {
            v24 = *(_QWORD *)(v18 + 48) + 16 * v23;
            LOWORD(v16) = *(_WORD *)v24;
            sub_2FE6CC0((__int64)v59, *(_QWORD *)a1, *(_QWORD *)(*(_QWORD *)(a1 + 8) + 64LL), v16, *(_QWORD *)(v24 + 8));
            switch ( v59[0] )
            {
              case 1:
                v46 = v16;
                sub_3844940(a1, v18, (unsigned int)v23);
                goto LABEL_20;
              case 2:
                v46 = v16;
                sub_3834970(a1, v18, (unsigned int)v23);
                goto LABEL_20;
              case 3:
                v46 = v16;
                sub_380B540(a1, v18, (unsigned int)v23);
                goto LABEL_20;
              case 4:
                v46 = v16;
                sub_3800480(a1, v18, (unsigned int)v23);
                goto LABEL_20;
              case 5:
                v46 = v16;
                sub_379A7A0(a1, v18, (unsigned int)v23);
                goto LABEL_20;
              case 6:
                v46 = v16;
                sub_37B35F0(a1, v18, (unsigned int)v23);
                goto LABEL_20;
              case 7:
                v46 = v16;
                sub_37B0610(a1, v18, (unsigned int)v23);
                goto LABEL_20;
              case 8:
                v46 = v16;
                sub_380ED90(a1, v18, (unsigned int)v23);
                goto LABEL_20;
              case 9:
                v46 = v16;
                sub_3811680(a1, v18, (unsigned int)v23);
LABEL_20:
                v47 = 1;
                goto LABEL_21;
              case 0xA:
LABEL_69:
                sub_C64ED0("Scalarization of scalable vectors is not supported.", 1u);
              default:
                if ( v44 != ++v23 )
                  continue;
                v46 = v16;
                break;
            }
            break;
          }
        }
      }
      if ( *(_DWORD *)(v18 + 64) )
      {
        v16 = v2;
        v33 = 0;
        v34 = *(unsigned int *)(v18 + 64);
        while ( 1 )
        {
          v35 = (unsigned int *)(*(_QWORD *)(v18 + 40) + 40 * v33);
          if ( *(_DWORD *)(*(_QWORD *)v35 + 24LL) != 9 && *(_DWORD *)(*(_QWORD *)v35 + 24LL) != 35 )
            break;
LABEL_50:
          if ( v34 == ++v33 )
          {
LABEL_51:
            v2 = v16;
            goto LABEL_21;
          }
        }
        v36 = *(_QWORD *)(*(_QWORD *)v35 + 48LL) + 16LL * v35[2];
        LOWORD(v16) = *(_WORD *)v36;
        sub_2FE6CC0((__int64)v59, *(_QWORD *)a1, *(_QWORD *)(*(_QWORD *)(a1 + 8) + 64LL), v16, *(_QWORD *)(v36 + 8));
        switch ( v59[0] )
        {
          case 0:
            goto LABEL_50;
          case 1:
            v2 = v16;
            v37 = sub_3845740(a1, v18, (unsigned int)v33);
            v47 = 1;
            goto LABEL_54;
          case 2:
            v2 = v16;
            v37 = sub_382B700(a1, v18, (unsigned int)v33);
            v47 = 1;
            goto LABEL_54;
          case 3:
            v2 = v16;
            v37 = sub_380A350(a1, v18, (unsigned int)v33);
            v47 = 1;
            goto LABEL_54;
          case 4:
            v2 = v16;
            v37 = sub_38033A0(a1, v18, (unsigned int)v33);
            v47 = 1;
            goto LABEL_54;
          case 5:
            v2 = v16;
            v37 = sub_379A4D0(a1, v18, (unsigned int)v33);
            v47 = 1;
            goto LABEL_54;
          case 6:
            v2 = v16;
            v37 = sub_378D7C0(a1, v18, (unsigned int)v33);
            v47 = 1;
            goto LABEL_54;
          case 7:
            v2 = v16;
            v37 = sub_37ADC40(a1, v18, (unsigned int)v33);
            v47 = 1;
            goto LABEL_54;
          case 8:
            v2 = v16;
            v37 = sub_380D6E0(a1, v18, (unsigned int)v33);
            v47 = 1;
            goto LABEL_54;
          case 9:
            v2 = v16;
            v37 = sub_3813430(a1, v18, (unsigned int)v33);
            v47 = 1;
LABEL_54:
            if ( !v37 )
              goto LABEL_21;
            *(_DWORD *)(v18 + 36) = -1;
            a2 = v18;
            v39 = sub_375EBD0(a1, v18, v38, v16, v6, v7);
            if ( (__int64 *)v18 != v39 && *(_DWORD *)(v18 + 68) )
            {
              v45 = v2;
              v40 = 0;
              v41 = *(_DWORD *)(v18 + 68);
              do
              {
                v42 = v40;
                v43 = v40;
                a2 = v18;
                ++v40;
                sub_3760E70(a1, v18, v42, (unsigned __int64)v39, v43);
              }
              while ( v40 != v41 );
              v2 = v45;
            }
            break;
          case 0xA:
            goto LABEL_69;
          default:
            goto LABEL_51;
        }
        goto LABEL_31;
      }
LABEL_21:
      v25 = *(_QWORD *)(v18 + 56);
      a2 = a1 + 1624;
      *(_DWORD *)(v18 + 36) = -3;
      if ( v25 )
        break;
LABEL_31:
      v11 = v58;
      *(_QWORD *)(v56 + 1024) = v58;
      if ( !*(_DWORD *)(a1 + 1616) )
        goto LABEL_32;
    }
    while ( 1 )
    {
      v26 = *(_QWORD *)(v25 + 16);
      v27 = *(_DWORD *)(v26 + 36);
      if ( v27 > 0 )
        goto LABEL_27;
      if ( v27 == -1 )
      {
LABEL_23:
        v25 = *(_QWORD *)(v25 + 32);
        if ( !v25 )
          goto LABEL_31;
      }
      else
      {
        v27 = *(_DWORD *)(v26 + 64);
LABEL_27:
        *(_DWORD *)(v26 + 36) = v27 - 1;
        if ( v27 != 1 )
          goto LABEL_23;
        v28 = *(unsigned int *)(a1 + 1616);
        v16 = *(unsigned int *)(a1 + 1620);
        if ( v28 + 1 > v16 )
        {
          a2 = a1 + 1624;
          sub_C8D5F0(a1 + 1608, (const void *)(a1 + 1624), v28 + 1, 8u, v6, v7);
          v28 = *(unsigned int *)(a1 + 1616);
        }
        *(_QWORD *)(*(_QWORD *)(a1 + 1608) + 8 * v28) = v26;
        ++*(_DWORD *)(a1 + 1616);
        v25 = *(_QWORD *)(v25 + 32);
        if ( !v25 )
          goto LABEL_31;
      }
    }
  }
LABEL_32:
  if ( (_BYTE)qword_5050F68 )
    sub_37594D0(a1, a2, v11, v16, v6, v7);
  v29 = v69;
  v30 = *(_QWORD *)(a1 + 8);
  v31 = v70;
  if ( v69 )
  {
    nullsub_1875();
    v52 = v31;
    v51 = v29;
    *(_QWORD *)(v30 + 384) = v29;
    *(_DWORD *)(v30 + 392) = v52;
    sub_33E2B60();
  }
  else
  {
    v50 = v70;
    v49 = 0;
    *(_QWORD *)(v30 + 384) = 0;
    *(_DWORD *)(v30 + 392) = v50;
  }
  sub_33F7860(*(const __m128i **)(a1 + 8));
  sub_33CF710((__int64)v60);
  return v47;
}
