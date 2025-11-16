// Function: sub_32EB790
// Address: 0x32eb790
//
__int64 __fastcall sub_32EB790(__int64 a1, __int64 a2, __int64 *a3, int a4, char a5)
{
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rdi
  __int64 v12; // r13
  __int64 v13; // r14
  _QWORD *v14; // rdi
  _QWORD *v15; // rsi
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // rbx
  int v19; // eax
  unsigned int v20; // esi
  unsigned int v21; // edi
  _QWORD *v22; // rcx
  __int64 v23; // rdx
  _QWORD *v24; // r11
  int v25; // eax
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // r8
  __int64 v29; // r9
  __int64 v30; // rax
  __int64 v32; // rax
  __int64 *v33; // r12
  __int64 v34; // r11
  __int64 *v35; // rbx
  unsigned int v36; // eax
  _QWORD *v37; // rdi
  __int64 v38; // rcx
  unsigned int v39; // esi
  int v40; // ecx
  int v41; // ecx
  unsigned int v42; // edx
  _QWORD *v43; // r10
  __int64 v44; // rdi
  int v45; // eax
  __int64 v46; // rax
  int v47; // eax
  int v48; // ecx
  int v49; // ecx
  unsigned int v50; // edx
  __int64 v51; // rdi
  int v52; // eax
  int v53; // ecx
  __int64 v54; // rdi
  unsigned int v55; // edx
  __int64 v56; // rsi
  int v57; // eax
  int v58; // edx
  __int64 v59; // rsi
  _QWORD *v60; // rdi
  unsigned int v61; // r12d
  __int64 v62; // rcx
  __int64 v63; // [rsp+8h] [rbp-98h]
  int v64; // [rsp+8h] [rbp-98h]
  __int64 v65; // [rsp+8h] [rbp-98h]
  int v66; // [rsp+8h] [rbp-98h]
  int v67; // [rsp+8h] [rbp-98h]
  const void *v68; // [rsp+10h] [rbp-90h]
  int v69; // [rsp+18h] [rbp-88h]
  __int64 v70; // [rsp+18h] [rbp-88h]
  __int64 v72; // [rsp+30h] [rbp-70h]
  __int64 *v73; // [rsp+38h] [rbp-68h]
  __int64 v74; // [rsp+48h] [rbp-58h] BYREF
  __int64 (__fastcall **v75)(); // [rsp+50h] [rbp-50h] BYREF
  __int64 v76; // [rsp+58h] [rbp-48h]
  __int64 v77; // [rsp+60h] [rbp-40h]
  __int64 v78; // [rsp+68h] [rbp-38h]

  v9 = *(_QWORD *)a1;
  v78 = a1;
  v10 = *(_QWORD *)(v9 + 768);
  v77 = v9;
  v76 = v10;
  *(_QWORD *)(v9 + 768) = &v75;
  v11 = *(_QWORD *)a1;
  v75 = off_4A360B8;
  sub_3415F70(v11, a2, a3);
  if ( a5 && a4 )
  {
    v73 = a3;
    v72 = (__int64)&a3[2 * (unsigned int)(a4 - 1) + 2];
    v68 = (const void *)(a1 + 56);
    while ( 1 )
    {
      v12 = *v73;
      if ( *v73 )
        break;
LABEL_33:
      v73 += 2;
      if ( (__int64 *)v72 == v73 )
        goto LABEL_34;
    }
    v13 = *(_QWORD *)(v12 + 56);
    if ( !v13 )
    {
LABEL_28:
      if ( *(_DWORD *)(v12 + 24) != 328 )
      {
        v74 = v12;
        sub_32B3B20(a1 + 568, &v74);
        if ( *(int *)(v12 + 88) < 0 )
        {
          *(_DWORD *)(v12 + 88) = *(_DWORD *)(a1 + 48);
          v30 = *(unsigned int *)(a1 + 48);
          if ( v30 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 52) )
          {
            sub_C8D5F0(a1 + 40, v68, v30 + 1, 8u, v28, v29);
            v30 = *(unsigned int *)(a1 + 48);
          }
          *(_QWORD *)(*(_QWORD *)(a1 + 40) + 8 * v30) = v12;
          ++*(_DWORD *)(a1 + 48);
        }
      }
      goto LABEL_33;
    }
    while ( 1 )
    {
      while ( 1 )
      {
        v18 = *(_QWORD *)(v13 + 16);
        if ( *(_DWORD *)(v18 + 24) == 328 )
          goto LABEL_9;
        v19 = *(_DWORD *)(a1 + 584);
        v74 = *(_QWORD *)(v13 + 16);
        if ( !v19 )
          break;
        v20 = *(_DWORD *)(a1 + 592);
        if ( !v20 )
        {
          ++*(_QWORD *)(a1 + 568);
LABEL_67:
          sub_32B3220(a1 + 568, 2 * v20);
          v52 = *(_DWORD *)(a1 + 592);
          if ( !v52 )
            goto LABEL_111;
          v53 = v52 - 1;
          v54 = *(_QWORD *)(a1 + 576);
          v55 = (v52 - 1) & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
          v24 = (_QWORD *)(v54 + 8LL * v55);
          v56 = *v24;
          v25 = *(_DWORD *)(a1 + 584) + 1;
          if ( v18 != *v24 )
          {
            v17 = 1;
            v16 = 0;
            while ( v56 != -4096 )
            {
              if ( !v16 && v56 == -8192 )
                v16 = (__int64)v24;
              v55 = v53 & (v17 + v55);
              v24 = (_QWORD *)(v54 + 8LL * v55);
              v56 = *v24;
              if ( v18 == *v24 )
                goto LABEL_20;
              v17 = (unsigned int)(v17 + 1);
            }
            if ( v16 )
              v24 = (_QWORD *)v16;
          }
          goto LABEL_20;
        }
        v17 = v20 - 1;
        v16 = *(_QWORD *)(a1 + 576);
        v21 = v17 & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
        v22 = (_QWORD *)(v16 + 8LL * v21);
        v23 = *v22;
        if ( v18 == *v22 )
          goto LABEL_8;
        v69 = 1;
        v24 = 0;
        while ( v23 != -4096 )
        {
          if ( v24 || v23 != -8192 )
            v22 = v24;
          v21 = v17 & (v69 + v21);
          v23 = *(_QWORD *)(v16 + 8LL * v21);
          if ( v18 == v23 )
            goto LABEL_8;
          ++v69;
          v24 = v22;
          v22 = (_QWORD *)(v16 + 8LL * v21);
        }
        if ( !v24 )
          v24 = v22;
        v25 = v19 + 1;
        ++*(_QWORD *)(a1 + 568);
        if ( 4 * v25 >= 3 * v20 )
          goto LABEL_67;
        if ( v20 - *(_DWORD *)(a1 + 588) - v25 <= v20 >> 3 )
        {
          sub_32B3220(a1 + 568, v20);
          v57 = *(_DWORD *)(a1 + 592);
          if ( !v57 )
          {
LABEL_111:
            ++*(_DWORD *)(a1 + 584);
            BUG();
          }
          v58 = v57 - 1;
          v59 = *(_QWORD *)(a1 + 576);
          v16 = 1;
          v60 = 0;
          v61 = (v57 - 1) & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
          v24 = (_QWORD *)(v59 + 8LL * v61);
          v62 = *v24;
          v25 = *(_DWORD *)(a1 + 584) + 1;
          if ( v18 != *v24 )
          {
            while ( v62 != -4096 )
            {
              if ( !v60 && v62 == -8192 )
                v60 = v24;
              v17 = (unsigned int)(v16 + 1);
              v61 = v58 & (v16 + v61);
              v24 = (_QWORD *)(v59 + 8LL * v61);
              v62 = *v24;
              if ( v18 == *v24 )
                goto LABEL_20;
              v16 = (unsigned int)v17;
            }
            if ( v60 )
              v24 = v60;
          }
        }
LABEL_20:
        *(_DWORD *)(a1 + 584) = v25;
        if ( *v24 != -4096 )
          --*(_DWORD *)(a1 + 588);
        *v24 = v18;
        v26 = *(unsigned int *)(a1 + 608);
        if ( v26 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 612) )
        {
          sub_C8D5F0(a1 + 600, (const void *)(a1 + 616), v26 + 1, 8u, v16, v17);
          v26 = *(unsigned int *)(a1 + 608);
        }
        *(_QWORD *)(*(_QWORD *)(a1 + 600) + 8 * v26) = v18;
        ++*(_DWORD *)(a1 + 608);
        if ( *(int *)(v18 + 88) < 0 )
          goto LABEL_25;
LABEL_9:
        v13 = *(_QWORD *)(v13 + 32);
        if ( !v13 )
          goto LABEL_28;
      }
      v14 = *(_QWORD **)(a1 + 600);
      v15 = &v14[*(unsigned int *)(a1 + 608)];
      if ( v15 == sub_325EB50(v14, (__int64)v15, &v74) )
      {
        if ( v16 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 612) )
        {
          sub_C8D5F0(a1 + 600, (const void *)(a1 + 616), v16 + 1, 8u, v16, v17);
          v15 = (_QWORD *)(*(_QWORD *)(a1 + 600) + 8LL * *(unsigned int *)(a1 + 608));
        }
        *v15 = v18;
        v32 = (unsigned int)(*(_DWORD *)(a1 + 608) + 1);
        *(_DWORD *)(a1 + 608) = v32;
        if ( (unsigned int)v32 > 0x20 )
          break;
      }
LABEL_8:
      if ( *(int *)(v18 + 88) >= 0 )
        goto LABEL_9;
LABEL_25:
      *(_DWORD *)(v18 + 88) = *(_DWORD *)(a1 + 48);
      v27 = *(unsigned int *)(a1 + 48);
      if ( v27 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 52) )
      {
        sub_C8D5F0(a1 + 40, v68, v27 + 1, 8u, v16, v17);
        v27 = *(unsigned int *)(a1 + 48);
      }
      *(_QWORD *)(*(_QWORD *)(a1 + 40) + 8 * v27) = v18;
      ++*(_DWORD *)(a1 + 48);
      v13 = *(_QWORD *)(v13 + 32);
      if ( !v13 )
        goto LABEL_28;
    }
    v33 = *(__int64 **)(a1 + 600);
    v70 = a1 + 568;
    v34 = v18;
    v35 = &v33[v32];
    while ( 1 )
    {
      v39 = *(_DWORD *)(a1 + 592);
      if ( !v39 )
        break;
      v17 = v39 - 1;
      v16 = *(_QWORD *)(a1 + 576);
      v36 = v17 & (((unsigned int)*v33 >> 9) ^ ((unsigned int)*v33 >> 4));
      v37 = (_QWORD *)(v16 + 8LL * v36);
      v38 = *v37;
      if ( *v33 == *v37 )
      {
LABEL_42:
        if ( v35 == ++v33 )
          goto LABEL_50;
      }
      else
      {
        v64 = 1;
        v43 = 0;
        while ( v38 != -4096 )
        {
          if ( v43 || v38 != -8192 )
            v37 = v43;
          v36 = v17 & (v64 + v36);
          v38 = *(_QWORD *)(v16 + 8LL * v36);
          if ( *v33 == v38 )
            goto LABEL_42;
          ++v64;
          v43 = v37;
          v37 = (_QWORD *)(v16 + 8LL * v36);
        }
        v47 = *(_DWORD *)(a1 + 584);
        if ( !v43 )
          v43 = v37;
        ++*(_QWORD *)(a1 + 568);
        v45 = v47 + 1;
        if ( 4 * v45 < 3 * v39 )
        {
          if ( v39 - *(_DWORD *)(a1 + 588) - v45 > v39 >> 3 )
            goto LABEL_47;
          v65 = v34;
          sub_32B3220(v70, v39);
          v48 = *(_DWORD *)(a1 + 592);
          if ( !v48 )
          {
LABEL_110:
            ++*(_DWORD *)(a1 + 584);
            BUG();
          }
          v49 = v48 - 1;
          v16 = *(_QWORD *)(a1 + 576);
          v34 = v65;
          v50 = v49 & (((unsigned int)*v33 >> 9) ^ ((unsigned int)*v33 >> 4));
          v43 = (_QWORD *)(v16 + 8LL * v50);
          v51 = *v43;
          v45 = *(_DWORD *)(a1 + 584) + 1;
          if ( *v33 == *v43 )
            goto LABEL_47;
          v66 = 1;
          v17 = 0;
          while ( v51 != -4096 )
          {
            if ( v51 == -8192 && !v17 )
              v17 = (__int64)v43;
            v50 = v49 & (v66 + v50);
            v43 = (_QWORD *)(v16 + 8LL * v50);
            v51 = *v43;
            if ( *v33 == *v43 )
              goto LABEL_47;
            ++v66;
          }
          goto LABEL_86;
        }
LABEL_45:
        v63 = v34;
        sub_32B3220(v70, 2 * v39);
        v40 = *(_DWORD *)(a1 + 592);
        if ( !v40 )
          goto LABEL_110;
        v41 = v40 - 1;
        v16 = *(_QWORD *)(a1 + 576);
        v34 = v63;
        v42 = v41 & (((unsigned int)*v33 >> 9) ^ ((unsigned int)*v33 >> 4));
        v43 = (_QWORD *)(v16 + 8LL * v42);
        v44 = *v43;
        v45 = *(_DWORD *)(a1 + 584) + 1;
        if ( *v33 == *v43 )
          goto LABEL_47;
        v67 = 1;
        v17 = 0;
        while ( v44 != -4096 )
        {
          if ( v44 == -8192 && !v17 )
            v17 = (__int64)v43;
          v42 = v41 & (v67 + v42);
          v43 = (_QWORD *)(v16 + 8LL * v42);
          v44 = *v43;
          if ( *v33 == *v43 )
            goto LABEL_47;
          ++v67;
        }
LABEL_86:
        if ( v17 )
          v43 = (_QWORD *)v17;
LABEL_47:
        *(_DWORD *)(a1 + 584) = v45;
        if ( *v43 != -4096 )
          --*(_DWORD *)(a1 + 588);
        v46 = *v33++;
        *v43 = v46;
        if ( v35 == v33 )
        {
LABEL_50:
          v18 = v34;
          goto LABEL_8;
        }
      }
    }
    ++*(_QWORD *)(a1 + 568);
    goto LABEL_45;
  }
LABEL_34:
  if ( !*(_QWORD *)(a2 + 56) )
    sub_32EB240(a1, a2);
  *(_QWORD *)(v77 + 768) = v76;
  return a2;
}
