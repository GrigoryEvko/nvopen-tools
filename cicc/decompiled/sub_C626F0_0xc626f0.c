// Function: sub_C626F0
// Address: 0xc626f0
//
__int64 __fastcall sub_C626F0(__int64 a1, __int64 a2)
{
  char *v3; // r12
  char *v4; // rbx
  __int64 v5; // r13
  _BYTE *v6; // rdi
  int v7; // edx
  char *v8; // rax
  __int64 v9; // rsi
  __int64 *v10; // rax
  __m128i *v11; // rdx
  __int64 v12; // r12
  __m128i si128; // xmm0
  __int64 v14; // r8
  int *v15; // rdx
  int v16; // r11d
  unsigned int v17; // edi
  int *v18; // rax
  int v19; // ecx
  __int64 v20; // rdx
  _WORD *v21; // rdx
  int v22; // ebx
  __int64 v23; // r13
  _BYTE *v24; // rsi
  __int64 v25; // rdx
  __int64 v26; // rax
  __int64 *v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // r13
  unsigned int v32; // esi
  __int64 v33; // r9
  int *v34; // rdx
  int v35; // r11d
  unsigned int v36; // r8d
  int *v37; // rax
  int v38; // edi
  __int64 v39; // rsi
  __int64 v40; // rdi
  _BYTE *v41; // rax
  unsigned int v42; // esi
  int v43; // ecx
  int v44; // ecx
  __int64 v45; // r8
  unsigned int v46; // esi
  int v47; // eax
  int v48; // edi
  int v49; // eax
  int v50; // eax
  int v51; // eax
  int v52; // ecx
  int v53; // ecx
  __int64 v54; // rdi
  int *v55; // r8
  unsigned int v56; // r13d
  int v57; // r10d
  int v58; // esi
  __int64 v59; // rax
  int v60; // ecx
  int v61; // ecx
  __int64 v62; // r8
  unsigned int v63; // esi
  int v64; // edi
  int v65; // r11d
  int *v66; // r10
  __int64 result; // rax
  int v68; // esi
  int v69; // esi
  __int64 v70; // r8
  int *v71; // r9
  int v72; // r11d
  unsigned int v73; // ecx
  int v74; // edi
  int v75; // r11d
  int *v76; // r9
  int v77; // [rsp+4h] [rbp-19Ch]
  char *v79; // [rsp+28h] [rbp-178h]
  char *v80; // [rsp+38h] [rbp-168h]
  __int64 v81[2]; // [rsp+40h] [rbp-160h] BYREF
  _QWORD v82[2]; // [rsp+50h] [rbp-150h] BYREF
  void *base; // [rsp+60h] [rbp-140h] BYREF
  __int64 v84; // [rsp+68h] [rbp-138h]
  _BYTE v85[304]; // [rsp+70h] [rbp-130h] BYREF

  v3 = *(char **)(a1 + 88);
  v4 = *(char **)(a1 + 80);
  base = v85;
  v84 = 0x1000000000LL;
  v5 = (v3 - v4) >> 5;
  if ( (unsigned __int64)(v3 - v4) > 0x200 )
  {
    sub_C8D5F0(&base, v85, (v3 - v4) >> 5, 16);
    v7 = v84;
    v6 = base;
    v8 = (char *)base + 16 * (unsigned int)v84;
  }
  else
  {
    v6 = v85;
    v7 = 0;
    v8 = v85;
  }
  if ( v4 != v3 )
  {
    do
    {
      if ( v8 )
      {
        *(_QWORD *)v8 = *(_QWORD *)v4;
        *((_QWORD *)v8 + 1) = *((_QWORD *)v4 + 1);
      }
      v4 += 32;
      v8 += 16;
    }
    while ( v3 != v4 );
    v6 = base;
    v7 = v84;
  }
  LODWORD(v84) = v5 + v7;
  v9 = 16LL * (unsigned int)(v5 + v7);
  if ( (unsigned int)(v5 + v7) > 1uLL )
  {
    v9 >>= 4;
    qsort(v6, v9, 0x10u, (__compar_fn_t)sub_A16990);
  }
  v10 = sub_C60B10();
  v11 = *(__m128i **)(a2 + 32);
  v12 = (__int64)v10;
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v11 <= 0x14u )
  {
    v9 = (__int64)"Counters and values:\n";
    sub_CB6200(a2, "Counters and values:\n", 21);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F66770);
    v11[1].m128i_i32[0] = 980641141;
    v11[1].m128i_i8[4] = 10;
    *v11 = si128;
    *(_QWORD *)(a2 + 32) += 21LL;
  }
  v79 = (char *)base + 16 * (unsigned int)v84;
  if ( base != v79 )
  {
    v80 = (char *)base;
    while ( 1 )
    {
      v22 = 0;
      v23 = 0x1FFFFFFFE0LL;
      v24 = *(_BYTE **)v80;
      v25 = *(_QWORD *)v80 + *((_QWORD *)v80 + 1);
      v81[0] = (__int64)v82;
      sub_C5F830(v81, v24, v25);
      v26 = sub_C61310(a1 + 32, (__int64)v81);
      if ( v26 != a1 + 40 )
      {
        v22 = *(_DWORD *)(v26 + 64);
        v23 = 32LL * (unsigned int)(v22 - 1);
      }
      if ( (_QWORD *)v81[0] != v82 )
        j_j___libc_free_0(v81[0], v82[0] + 1LL);
      v27 = (__int64 *)(v23 + *(_QWORD *)(a1 + 80));
      v28 = *v27;
      v81[1] = v27[1];
      v81[0] = v28;
      v82[0] = 0x100000020LL;
      v29 = sub_CB6A20(a2, v81);
      v30 = *(_QWORD *)(v29 + 32);
      v31 = v29;
      if ( (unsigned __int64)(*(_QWORD *)(v29 + 24) - v30) <= 2 )
      {
        v59 = sub_CB6200(v29, ": {", 3);
        v32 = *(_DWORD *)(v12 + 24);
        v31 = v59;
        if ( !v32 )
        {
LABEL_67:
          ++*(_QWORD *)v12;
          goto LABEL_68;
        }
      }
      else
      {
        *(_BYTE *)(v30 + 2) = 123;
        *(_WORD *)v30 = 8250;
        *(_QWORD *)(v29 + 32) += 3LL;
        v32 = *(_DWORD *)(v12 + 24);
        if ( !v32 )
          goto LABEL_67;
      }
      v33 = *(_QWORD *)(v12 + 8);
      v34 = 0;
      v35 = 1;
      v36 = (v32 - 1) & (37 * v22);
      v37 = (int *)(v33 + ((unsigned __int64)v36 << 7));
      v38 = *v37;
      if ( v22 == *v37 )
      {
LABEL_26:
        v39 = *((_QWORD *)v37 + 1);
        goto LABEL_27;
      }
      while ( v38 != -1 )
      {
        if ( !v34 && v38 == -2 )
          v34 = v37;
        v36 = (v32 - 1) & (v35 + v36);
        v37 = (int *)(v33 + ((unsigned __int64)v36 << 7));
        v38 = *v37;
        if ( v22 == *v37 )
          goto LABEL_26;
        ++v35;
      }
      if ( !v34 )
        v34 = v37;
      v49 = *(_DWORD *)(v12 + 16);
      ++*(_QWORD *)v12;
      v50 = v49 + 1;
      if ( 4 * v50 < 3 * v32 )
      {
        if ( v32 - *(_DWORD *)(v12 + 20) - v50 <= v32 >> 3 )
        {
          v77 = 37 * v22;
          sub_C61E30(v12, v32);
          v68 = *(_DWORD *)(v12 + 24);
          if ( !v68 )
          {
LABEL_115:
            ++*(_DWORD *)(v12 + 16);
            BUG();
          }
          v69 = v68 - 1;
          v70 = *(_QWORD *)(v12 + 8);
          v71 = 0;
          v72 = 1;
          v50 = *(_DWORD *)(v12 + 16) + 1;
          v73 = v69 & v77;
          v34 = (int *)(v70 + ((unsigned __int64)(v69 & (unsigned int)v77) << 7));
          v74 = *v34;
          if ( v22 != *v34 )
          {
            while ( v74 != -1 )
            {
              if ( v74 != -2 || v71 )
                v34 = v71;
              v73 = v69 & (v72 + v73);
              v74 = *(_DWORD *)(v70 + ((unsigned __int64)v73 << 7));
              if ( v22 == v74 )
              {
                v34 = (int *)(v70 + ((unsigned __int64)v73 << 7));
                goto LABEL_46;
              }
              ++v72;
              v71 = v34;
              v34 = (int *)(v70 + ((unsigned __int64)v73 << 7));
            }
            if ( v71 )
              v34 = v71;
          }
        }
        goto LABEL_46;
      }
LABEL_68:
      sub_C61E30(v12, 2 * v32);
      v60 = *(_DWORD *)(v12 + 24);
      if ( !v60 )
        goto LABEL_115;
      v61 = v60 - 1;
      v62 = *(_QWORD *)(v12 + 8);
      v63 = v61 & (37 * v22);
      v50 = *(_DWORD *)(v12 + 16) + 1;
      v34 = (int *)(v62 + ((unsigned __int64)v63 << 7));
      v64 = *v34;
      if ( v22 != *v34 )
      {
        v65 = 1;
        v66 = 0;
        while ( v64 != -1 )
        {
          if ( !v66 && v64 == -2 )
            v66 = v34;
          v63 = v61 & (v65 + v63);
          v34 = (int *)(v62 + ((unsigned __int64)v63 << 7));
          v64 = *v34;
          if ( v22 == *v34 )
            goto LABEL_46;
          ++v65;
        }
        if ( v66 )
          v34 = v66;
      }
LABEL_46:
      *(_DWORD *)(v12 + 16) = v50;
      if ( *v34 != -1 )
        --*(_DWORD *)(v12 + 20);
      *v34 = v22;
      memset(v34 + 2, 0, 0x78u);
      v39 = 0;
      *((_QWORD *)v34 + 4) = v34 + 12;
      *((_QWORD *)v34 + 8) = v34 + 20;
      *((_QWORD *)v34 + 9) = 0x300000000LL;
LABEL_27:
      v40 = sub_CB59F0(v31, v39);
      v41 = *(_BYTE **)(v40 + 32);
      if ( *(_BYTE **)(v40 + 24) == v41 )
      {
        sub_CB6200(v40, ",", 1);
      }
      else
      {
        *v41 = 44;
        ++*(_QWORD *)(v40 + 32);
      }
      v42 = *(_DWORD *)(v12 + 24);
      if ( !v42 )
      {
        ++*(_QWORD *)v12;
        goto LABEL_31;
      }
      v14 = *(_QWORD *)(v12 + 8);
      v15 = 0;
      v16 = 1;
      v17 = (v42 - 1) & (37 * v22);
      v18 = (int *)(v14 + ((unsigned __int64)v17 << 7));
      v19 = *v18;
      if ( v22 != *v18 )
      {
        while ( v19 != -1 )
        {
          if ( !v15 && v19 == -2 )
            v15 = v18;
          v17 = (v42 - 1) & (v16 + v17);
          v18 = (int *)(v14 + ((unsigned __int64)v17 << 7));
          v19 = *v18;
          if ( v22 == *v18 )
            goto LABEL_15;
          ++v16;
        }
        if ( !v15 )
          v15 = v18;
        v51 = *(_DWORD *)(v12 + 16);
        ++*(_QWORD *)v12;
        v47 = v51 + 1;
        if ( 4 * v47 >= 3 * v42 )
        {
LABEL_31:
          sub_C61E30(v12, 2 * v42);
          v43 = *(_DWORD *)(v12 + 24);
          if ( !v43 )
            goto LABEL_114;
          v44 = v43 - 1;
          v45 = *(_QWORD *)(v12 + 8);
          v46 = v44 & (37 * v22);
          v47 = *(_DWORD *)(v12 + 16) + 1;
          v15 = (int *)(v45 + ((unsigned __int64)v46 << 7));
          v48 = *v15;
          if ( *v15 != v22 )
          {
            v75 = 1;
            v76 = 0;
            while ( v48 != -1 )
            {
              if ( !v76 && v48 == -2 )
                v76 = v15;
              v46 = v44 & (v75 + v46);
              v15 = (int *)(v45 + ((unsigned __int64)v46 << 7));
              v48 = *v15;
              if ( v22 == *v15 )
                goto LABEL_33;
              ++v75;
            }
            if ( v76 )
              v15 = v76;
          }
        }
        else if ( v42 - *(_DWORD *)(v12 + 20) - v47 <= v42 >> 3 )
        {
          sub_C61E30(v12, v42);
          v52 = *(_DWORD *)(v12 + 24);
          if ( !v52 )
          {
LABEL_114:
            ++*(_DWORD *)(v12 + 16);
            BUG();
          }
          v53 = v52 - 1;
          v54 = *(_QWORD *)(v12 + 8);
          v55 = 0;
          v56 = v53 & (37 * v22);
          v57 = 1;
          v47 = *(_DWORD *)(v12 + 16) + 1;
          v15 = (int *)(v54 + ((unsigned __int64)v56 << 7));
          v58 = *v15;
          if ( v22 != *v15 )
          {
            while ( v58 != -1 )
            {
              if ( v58 == -2 && !v55 )
                v55 = v15;
              v56 = v53 & (v57 + v56);
              v15 = (int *)(v54 + ((unsigned __int64)v56 << 7));
              v58 = *v15;
              if ( v22 == *v15 )
                goto LABEL_33;
              ++v57;
            }
            if ( v55 )
              v15 = v55;
          }
        }
LABEL_33:
        *(_DWORD *)(v12 + 16) = v47;
        if ( *v15 != -1 )
          --*(_DWORD *)(v12 + 20);
        *v15 = v22;
        memset(v15 + 2, 0, 0x78u);
        v9 = (__int64)(v15 + 20);
        *((_QWORD *)v15 + 4) = v15 + 12;
        *((_QWORD *)v15 + 8) = v15 + 20;
        *((_QWORD *)v15 + 9) = 0x300000000LL;
        v20 = 0;
        goto LABEL_16;
      }
LABEL_15:
      v9 = *((_QWORD *)v18 + 8);
      v20 = (unsigned int)v18[18];
LABEL_16:
      sub_C605E0(a2, (const __m128i *)v9, v20);
      v21 = *(_WORD **)(a2 + 32);
      if ( *(_QWORD *)(a2 + 24) - (_QWORD)v21 <= 1u )
      {
        v9 = (__int64)"}\n";
        sub_CB6200(a2, "}\n", 2);
      }
      else
      {
        *v21 = 2685;
        *(_QWORD *)(a2 + 32) += 2LL;
      }
      v80 += 16;
      if ( v79 == v80 )
      {
        v79 = (char *)base;
        break;
      }
    }
  }
  result = (__int64)v79;
  if ( v79 != v85 )
    return _libc_free(v79, v9);
  return result;
}
