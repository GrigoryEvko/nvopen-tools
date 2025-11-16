// Function: sub_2599D20
// Address: 0x2599d20
//
__int64 __fastcall sub_2599D20(__int64 a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 *v9; // r13
  __int64 *v10; // rbx
  __int64 *v11; // rsi
  unsigned __int64 *v12; // rbx
  unsigned int v13; // r13d
  __int64 v14; // r13
  __int64 v15; // rdi
  _BYTE *v16; // r12
  __int64 v17; // rcx
  __int64 v18; // rax
  _BYTE *v19; // rax
  __int64 v20; // rbx
  __int64 v21; // r12
  int v22; // edx
  unsigned int v23; // eax
  __int64 v24; // rsi
  __int64 v25; // r8
  __int64 v26; // rcx
  int v27; // eax
  int v28; // eax
  __int64 v29; // rcx
  _QWORD *v30; // rax
  __int64 *v31; // rsi
  __int64 v32; // rdi
  _QWORD *v33; // rdx
  char v35; // al
  __int64 *v36; // r8
  unsigned int v37; // esi
  int v38; // eax
  _QWORD *v39; // rdx
  int v40; // eax
  __int64 v41; // rax
  __int64 v42; // r13
  int v43; // edi
  __int64 v44; // rax
  __int64 *v45; // r13
  __int64 v46; // r11
  __int64 *v47; // r12
  __int64 v48; // r8
  unsigned int v49; // eax
  __int64 *v50; // rdi
  __int64 v51; // rcx
  unsigned int v52; // esi
  __int64 *v53; // r10
  int v54; // edx
  int v55; // esi
  _QWORD *v56; // rdi
  _QWORD *v57; // rsi
  __int64 v58; // rdx
  __int64 v59; // rcx
  __int64 v60; // r8
  __int64 v61; // r9
  int v62; // eax
  __int64 v63; // rdx
  __int64 v64; // rcx
  __int64 v65; // r8
  __int64 v66; // r9
  __int64 v67; // r12
  __int64 v68; // rax
  __int64 *v69; // rbx
  __int64 *v70; // r14
  __int64 *v71; // rdx
  __int64 v72; // [rsp+0h] [rbp-90h]
  bool v73; // [rsp+8h] [rbp-88h]
  __int64 v74; // [rsp+8h] [rbp-88h]
  int v75; // [rsp+8h] [rbp-88h]
  unsigned __int64 *v76; // [rsp+18h] [rbp-78h]
  __int64 v77; // [rsp+18h] [rbp-78h]
  __int64 v78; // [rsp+18h] [rbp-78h]
  __int64 v79; // [rsp+20h] [rbp-70h] BYREF
  _QWORD *v80; // [rsp+28h] [rbp-68h] BYREF
  __m128i v81; // [rsp+30h] [rbp-60h] BYREF
  _BYTE *v82; // [rsp+40h] [rbp-50h]
  char v83; // [rsp+50h] [rbp-40h]

  v4 = sub_25096F0((_QWORD *)(a1 + 72));
  v72 = v4;
  if ( *(_DWORD *)(a1 + 376) )
  {
    v73 = 0;
    if ( (*(_BYTE *)(v4 + 2) & 8) != 0 )
      v73 = !sub_B2AC90(v4);
    v9 = *(__int64 **)(a1 + 136);
    v10 = &v9[*(unsigned int *)(a1 + 144)];
    while ( v10 != v9 )
    {
      v11 = v9++;
      sub_2574950(a1 + 216, v11, v5, v6, v7, v8);
    }
    v12 = *(unsigned __int64 **)(a1 + 248);
    v76 = &v12[*(unsigned int *)(a1 + 256)];
    if ( v12 != v76 )
    {
      v13 = 1;
      while ( 1 )
      {
        v16 = (_BYTE *)*v12;
        if ( (unsigned __int8)(*(_BYTE *)*v12 - 34) > 0x33u )
          goto LABEL_14;
        v17 = 0x8000000000041LL;
        if ( !_bittest64(&v17, (unsigned int)*(unsigned __int8 *)*v12 - 34) )
          goto LABEL_14;
        sub_250D230((unsigned __int64 *)&v81, *v12, 5, 0);
        if ( !(unsigned __int8)sub_2599C30(a2, a1, &v81, 1, &v80, 0, 0) )
          break;
        if ( *v16 == 34 )
          goto LABEL_10;
        v18 = *((_QWORD *)v16 + 4);
        v14 = a2 + 2688;
        if ( v18 == *((_QWORD *)v16 + 5) + 48LL || !v18 )
        {
          v81 = (__m128i)4uLL;
          v82 = 0;
          goto LABEL_13;
        }
        v19 = (_BYTE *)(v18 - 24);
        v81 = (__m128i)4uLL;
        v82 = v19;
        if ( v19 == (_BYTE *)-8192LL || v19 == (_BYTE *)-4096LL )
          goto LABEL_13;
LABEL_12:
        sub_BD73F0((__int64)&v81);
LABEL_13:
        v15 = v14;
        v13 = 0;
        sub_25703E0(v15, v81.m128i_i8);
        sub_D68D70(&v81);
LABEL_14:
        if ( v76 == ++v12 )
          goto LABEL_24;
      }
      if ( v73 || *v16 != 34 )
        goto LABEL_14;
LABEL_10:
      v81 = (__m128i)4uLL;
      v14 = a2 + 3120;
      v82 = v16;
      if ( v16 == (_BYTE *)-4096LL || v16 == (_BYTE *)-8192LL )
        goto LABEL_13;
      goto LABEL_12;
    }
    v13 = 1;
LABEL_24:
    if ( !byte_4FEF5A8 && (unsigned int)sub_2207590((__int64)&byte_4FEF5A8) )
      sub_2207640((__int64)&byte_4FEF5A8);
    v20 = *(_QWORD *)(v72 + 80);
    if ( v72 + 72 != v20 )
    {
      v21 = v72 + 72;
      while ( 1 )
      {
        v25 = v20 - 24;
        v26 = *(_QWORD *)(a1 + 368);
        if ( !v20 )
          v25 = 0;
        v27 = *(_DWORD *)(a1 + 384);
        if ( !v27 )
          goto LABEL_32;
        v22 = v27 - 1;
        v23 = (v27 - 1) & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
        v24 = *(_QWORD *)(v26 + 8LL * v23);
        if ( v25 == v24 )
        {
LABEL_28:
          v20 = *(_QWORD *)(v20 + 8);
          if ( v21 == v20 )
            return v13;
        }
        else
        {
          v43 = 1;
          while ( v24 != -4096 )
          {
            v8 = (unsigned int)(v43 + 1);
            v23 = v22 & (v43 + v23);
            v24 = *(_QWORD *)(v26 + 8LL * v23);
            if ( v25 == v24 )
              goto LABEL_28;
            ++v43;
          }
LABEL_32:
          v28 = *(_DWORD *)(a2 + 3784);
          v79 = v25;
          if ( v28 )
          {
            v35 = sub_D6B660(a2 + 3768, &v79, &v80);
            v36 = &v79;
            if ( v35 )
              goto LABEL_41;
            v37 = *(_DWORD *)(a2 + 3792);
            v38 = *(_DWORD *)(a2 + 3784);
            v39 = v80;
            ++*(_QWORD *)(a2 + 3768);
            v40 = v38 + 1;
            v8 = 2 * v37;
            v81.m128i_i64[0] = (__int64)v39;
            if ( 4 * v40 >= 3 * v37 )
            {
              v37 *= 2;
            }
            else if ( v37 - *(_DWORD *)(a2 + 3788) - v40 > v37 >> 3 )
            {
LABEL_46:
              *(_DWORD *)(a2 + 3784) = v40;
              if ( *v39 != -4096 )
                --*(_DWORD *)(a2 + 3788);
              *v39 = v79;
              v41 = *(unsigned int *)(a2 + 3808);
              v42 = v79;
              if ( v41 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 3812) )
              {
                sub_C8D5F0(a2 + 3800, (const void *)(a2 + 3816), v41 + 1, 8u, (__int64)v36, v8);
                v41 = *(unsigned int *)(a2 + 3808);
              }
              *(_QWORD *)(*(_QWORD *)(a2 + 3800) + 8 * v41) = v42;
              ++*(_DWORD *)(a2 + 3808);
              goto LABEL_41;
            }
            sub_CF28B0(a2 + 3768, v37);
            sub_D6B660(a2 + 3768, &v79, &v81);
            v39 = (_QWORD *)v81.m128i_i64[0];
            v40 = *(_DWORD *)(a2 + 3784) + 1;
            goto LABEL_46;
          }
          v29 = *(unsigned int *)(a2 + 3808);
          v30 = *(_QWORD **)(a2 + 3800);
          v31 = &v30[v29];
          v32 = (8 * v29) >> 3;
          if ( (8 * v29) >> 5 )
          {
            v33 = &v30[4 * ((8 * v29) >> 5)];
            while ( v25 != *v30 )
            {
              if ( v25 == v30[1] )
              {
                ++v30;
                goto LABEL_40;
              }
              if ( v25 == v30[2] )
              {
                v30 += 2;
                goto LABEL_40;
              }
              if ( v25 == v30[3] )
              {
                v30 += 3;
                goto LABEL_40;
              }
              v30 += 4;
              if ( v33 == v30 )
              {
                v32 = v31 - v30;
                goto LABEL_58;
              }
            }
            goto LABEL_40;
          }
LABEL_58:
          if ( v32 == 2 )
            goto LABEL_81;
          if ( v32 != 3 )
          {
            if ( v32 != 1 )
              goto LABEL_61;
LABEL_83:
            if ( v25 != *v30 )
              goto LABEL_61;
            goto LABEL_40;
          }
          if ( v25 != *v30 )
          {
            ++v30;
LABEL_81:
            if ( v25 != *v30 )
            {
              ++v30;
              goto LABEL_83;
            }
          }
LABEL_40:
          if ( v31 != v30 )
            goto LABEL_41;
LABEL_61:
          if ( v29 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 3812) )
          {
            v78 = v25;
            sub_C8D5F0(a2 + 3800, (const void *)(a2 + 3816), v29 + 1, 8u, v25, v8);
            v25 = v78;
            v31 = (__int64 *)(*(_QWORD *)(a2 + 3800) + 8LL * *(unsigned int *)(a2 + 3808));
          }
          *v31 = v25;
          v44 = (unsigned int)(*(_DWORD *)(a2 + 3808) + 1);
          *(_DWORD *)(a2 + 3808) = v44;
          if ( (unsigned int)v44 > 8 )
          {
            v45 = *(__int64 **)(a2 + 3800);
            v77 = a2 + 3768;
            v46 = v21;
            v47 = &v45[v44];
            while ( 1 )
            {
              v52 = *(_DWORD *)(a2 + 3792);
              if ( !v52 )
                break;
              v8 = v52 - 1;
              v48 = *(_QWORD *)(a2 + 3776);
              v49 = v8 & (((unsigned int)*v45 >> 9) ^ ((unsigned int)*v45 >> 4));
              v50 = (__int64 *)(v48 + 8LL * v49);
              v51 = *v50;
              if ( *v45 != *v50 )
              {
                v75 = 1;
                v53 = 0;
                while ( v51 != -4096 )
                {
                  if ( v53 || v51 != -8192 )
                    v50 = v53;
                  v49 = v8 & (v75 + v49);
                  v51 = *(_QWORD *)(v48 + 8LL * v49);
                  if ( *v45 == v51 )
                    goto LABEL_66;
                  ++v75;
                  v53 = v50;
                  v50 = (__int64 *)(v48 + 8LL * v49);
                }
                v62 = *(_DWORD *)(a2 + 3784);
                if ( !v53 )
                  v53 = v50;
                ++*(_QWORD *)(a2 + 3768);
                v54 = v62 + 1;
                v81.m128i_i64[0] = (__int64)v53;
                if ( 4 * (v62 + 1) < 3 * v52 )
                {
                  if ( v52 - *(_DWORD *)(a2 + 3788) - v54 > v52 >> 3 )
                    goto LABEL_93;
                  v74 = v46;
LABEL_70:
                  sub_CF28B0(v77, v52);
                  sub_D6B660(v77, v45, &v81);
                  v53 = (__int64 *)v81.m128i_i64[0];
                  v46 = v74;
                  v54 = *(_DWORD *)(a2 + 3784) + 1;
LABEL_93:
                  *(_DWORD *)(a2 + 3784) = v54;
                  if ( *v53 != -4096 )
                    --*(_DWORD *)(a2 + 3788);
                  *v53 = *v45;
                  goto LABEL_66;
                }
LABEL_69:
                v74 = v46;
                v52 *= 2;
                goto LABEL_70;
              }
LABEL_66:
              if ( v47 == ++v45 )
              {
                v21 = v46;
                goto LABEL_41;
              }
            }
            ++*(_QWORD *)(a2 + 3768);
            v81.m128i_i64[0] = 0;
            goto LABEL_69;
          }
LABEL_41:
          v20 = *(_QWORD *)(v20 + 8);
          v13 = 0;
          if ( v21 == v20 )
            return v13;
        }
      }
    }
  }
  else
  {
    v13 = 0;
    if ( *(_BYTE *)(a2 + 4297) )
    {
      v55 = *(_DWORD *)(a2 + 3672);
      v80 = (_QWORD *)v4;
      if ( v55 )
      {
        sub_2571760((__int64)&v81, a2 + 3656, (__int64 *)&v80);
        if ( v83 )
          sub_255C0E0(a2 + 3688, (__int64)v80, v63, v64, v65, v66);
      }
      else
      {
        v56 = *(_QWORD **)(a2 + 3688);
        v57 = &v56[*(unsigned int *)(a2 + 3696)];
        if ( v57 == sub_2538080(v56, (__int64)v57, (__int64 *)&v80) )
        {
          v67 = a2 + 3656;
          sub_255C0E0(a2 + 3688, v72, v58, v59, v60, v61);
          v68 = *(unsigned int *)(a2 + 3696);
          if ( (unsigned int)v68 > 8 )
          {
            v69 = *(__int64 **)(a2 + 3688);
            v70 = &v69[v68];
            do
            {
              v71 = v69++;
              sub_2571760((__int64)&v81, v67, v71);
            }
            while ( v70 != v69 );
          }
        }
      }
      return 0;
    }
  }
  return v13;
}
