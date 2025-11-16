// Function: sub_1F1A750
// Address: 0x1f1a750
//
__int64 __fastcall sub_1F1A750(__int64 a1, int a2, int *a3, __int64 a4, char a5)
{
  int v5; // r9d
  __int64 v9; // r14
  unsigned __int64 v10; // rcx
  int v11; // r10d
  unsigned int v12; // edx
  __int64 v13; // r8
  __int64 v14; // rax
  __int64 v15; // r13
  __int64 v16; // rax
  int v17; // r8d
  int v18; // r9d
  __int64 v19; // r14
  __int64 v20; // rax
  unsigned __int64 v21; // rcx
  __int64 v22; // rdx
  unsigned int v23; // esi
  int v24; // eax
  unsigned __int64 v25; // rdi
  unsigned __int64 v26; // rcx
  unsigned __int64 v27; // rbx
  unsigned int i; // edi
  __int64 v29; // r15
  int v30; // ecx
  unsigned int v31; // edi
  unsigned int v32; // r13d
  __int64 v33; // rsi
  __int64 v34; // rsi
  char v35; // cl
  unsigned __int64 v36; // rsi
  __int64 v38; // rdx
  _QWORD *v39; // rdi
  _QWORD *v40; // rdx
  __int64 v41; // rcx
  int v42; // edi
  int v43; // edi
  int v44; // esi
  int v45; // esi
  int v46; // r11d
  __int64 v47; // r8
  __int64 v48; // r10
  unsigned __int64 v49; // rdi
  unsigned __int64 v50; // rdi
  unsigned int j; // ecx
  int v52; // edi
  unsigned int v53; // ecx
  int v54; // ecx
  int v55; // ecx
  int v56; // r10d
  __int64 v57; // rsi
  unsigned int v58; // ebx
  __int64 v59; // r8
  int v60; // edi
  unsigned int v61; // ebx
  __int64 v62; // [rsp+0h] [rbp-60h]
  int v63; // [rsp+0h] [rbp-60h]
  int v64; // [rsp+0h] [rbp-60h]
  __int64 v65; // [rsp+8h] [rbp-58h]
  int v66; // [rsp+8h] [rbp-58h]
  int v67; // [rsp+8h] [rbp-58h]
  int v68; // [rsp+10h] [rbp-50h]
  __int64 v69; // [rsp+10h] [rbp-50h]
  __int64 v70; // [rsp+10h] [rbp-50h]
  int v72; // [rsp+20h] [rbp-40h]
  unsigned __int64 v73; // [rsp+20h] [rbp-40h]
  __int64 v74; // [rsp+20h] [rbp-40h]
  int v75; // [rsp+20h] [rbp-40h]
  int v76; // [rsp+28h] [rbp-38h]
  __int64 v77; // [rsp+28h] [rbp-38h]
  int v78; // [rsp+28h] [rbp-38h]

  v5 = a2;
  v9 = *(_QWORD *)(a1 + 16);
  v10 = *(unsigned int *)(v9 + 408);
  v11 = *(_DWORD *)(**(_QWORD **)(*(_QWORD *)(a1 + 72) + 16LL)
                  + 4LL * (unsigned int)(*(_DWORD *)(*(_QWORD *)(a1 + 72) + 64LL) + a2));
  v12 = v11 & 0x7FFFFFFF;
  v13 = v11 & 0x7FFFFFFF;
  v14 = 8 * v13;
  if ( (v11 & 0x7FFFFFFFu) >= (unsigned int)v10 || (v15 = *(_QWORD *)(*(_QWORD *)(v9 + 400) + 8LL * v12)) == 0 )
  {
    v32 = v12 + 1;
    if ( (unsigned int)v10 < v12 + 1 )
    {
      v38 = v32;
      if ( v32 < v10 )
      {
        *(_DWORD *)(v9 + 408) = v32;
      }
      else if ( v32 > v10 )
      {
        if ( v32 > (unsigned __int64)*(unsigned int *)(v9 + 412) )
        {
          v62 = v11 & 0x7FFFFFFF;
          v75 = *(_DWORD *)(**(_QWORD **)(*(_QWORD *)(a1 + 72) + 16LL)
                          + 4LL * (unsigned int)(*(_DWORD *)(*(_QWORD *)(a1 + 72) + 64LL) + a2));
          sub_16CD150(v9 + 400, (const void *)(v9 + 416), v32, 8, v13, a2);
          v10 = *(unsigned int *)(v9 + 408);
          v13 = v62;
          v5 = a2;
          v14 = 8 * v62;
          v11 = v75;
          v38 = v32;
        }
        v33 = *(_QWORD *)(v9 + 400);
        v39 = (_QWORD *)(v33 + 8 * v38);
        v40 = (_QWORD *)(v33 + 8 * v10);
        v41 = *(_QWORD *)(v9 + 416);
        if ( v39 != v40 )
        {
          do
            *v40++ = v41;
          while ( v39 != v40 );
          v33 = *(_QWORD *)(v9 + 400);
        }
        *(_DWORD *)(v9 + 408) = v32;
        goto LABEL_19;
      }
    }
    v33 = *(_QWORD *)(v9 + 400);
LABEL_19:
    v78 = v5;
    v74 = v13;
    *(_QWORD *)(v33 + v14) = sub_1DBA290(v11);
    v15 = *(_QWORD *)(*(_QWORD *)(v9 + 400) + 8 * v74);
    sub_1DBB110((_QWORD *)v9, v15);
    v9 = *(_QWORD *)(a1 + 16);
    v5 = v78;
  }
  v72 = v5;
  v76 = *(_DWORD *)(v15 + 72);
  v16 = sub_145CBF0((__int64 *)(v9 + 296), 16, 16);
  v18 = v72;
  *(_QWORD *)(v16 + 8) = a4;
  v19 = v16;
  *(_DWORD *)v16 = v76;
  v20 = *(unsigned int *)(v15 + 72);
  if ( (unsigned int)v20 >= *(_DWORD *)(v15 + 76) )
  {
    sub_16CD150(v15 + 64, (const void *)(v15 + 80), 0, 8, v17, v72);
    v20 = *(unsigned int *)(v15 + 72);
    v18 = v72;
  }
  v21 = 0;
  *(_QWORD *)(*(_QWORD *)(v15 + 64) + 8 * v20) = v19;
  v22 = *(_QWORD *)(v15 + 104);
  ++*(_DWORD *)(v15 + 72);
  if ( !v22 )
    v21 = v19 & 0xFFFFFFFFFFFFFFFBLL;
  v23 = *(_DWORD *)(a1 + 424);
  v77 = 4LL * (v22 != 0);
  v24 = *a3;
  v73 = v21;
  if ( !v23 )
  {
    ++*(_QWORD *)(a1 + 400);
    goto LABEL_47;
  }
  v68 = 1;
  v65 = 0;
  v25 = ((((unsigned int)(37 * v24) | ((unsigned __int64)(unsigned int)(37 * v18) << 32))
        - 1
        - ((unsigned __int64)(unsigned int)(37 * v24) << 32)) >> 22)
      ^ (((unsigned int)(37 * v24) | ((unsigned __int64)(unsigned int)(37 * v18) << 32))
       - 1
       - ((unsigned __int64)(unsigned int)(37 * v24) << 32));
  v26 = ((v25 - 1 - (v25 << 13)) >> 8) ^ (v25 - 1 - (v25 << 13));
  v27 = (((((9 * v26) >> 15) ^ (9 * v26)) - 1 - ((((9 * v26) >> 15) ^ (9 * v26)) << 27)) >> 31)
      ^ ((((9 * v26) >> 15) ^ (9 * v26)) - 1 - ((((9 * v26) >> 15) ^ (9 * v26)) << 27));
  for ( i = v27 & (v23 - 1); ; i = (v23 - 1) & v31 )
  {
    v29 = *(_QWORD *)(a1 + 408) + 16LL * i;
    v30 = *(_DWORD *)v29;
    if ( v18 == *(_DWORD *)v29 && v24 == *(_DWORD *)(v29 + 4) )
    {
      v34 = *(_QWORD *)(v29 + 8);
LABEL_22:
      v35 = a5;
      v36 = v34 & 0xFFFFFFFFFFFFFFF8LL;
      if ( v36 )
      {
        sub_1F15470((_QWORD *)a1, v15, v36, a5);
        v35 = a5;
        *(_QWORD *)(v29 + 8) = v77;
      }
      sub_1F15470((_QWORD *)a1, v15, v19, v35);
      return v19;
    }
    if ( v30 == -1 )
      break;
    if ( v30 == -2 && *(_DWORD *)(v29 + 4) == -2 )
    {
      if ( v65 )
        v29 = v65;
      v65 = v29;
    }
LABEL_16:
    v31 = v68 + i;
    ++v68;
  }
  if ( *(_DWORD *)(v29 + 4) != -1 )
    goto LABEL_16;
  if ( v65 )
    v29 = v65;
  v42 = *(_DWORD *)(a1 + 416);
  ++*(_QWORD *)(a1 + 400);
  v43 = v42 + 1;
  if ( 4 * v43 >= 3 * v23 )
  {
LABEL_47:
    v63 = v18;
    v66 = v24;
    v69 = v22;
    sub_1F1A4C0(a1 + 400, 2 * v23);
    v44 = *(_DWORD *)(a1 + 424);
    if ( v44 )
    {
      v18 = v63;
      v24 = v66;
      v45 = v44 - 1;
      v46 = 1;
      v47 = *(_QWORD *)(a1 + 408);
      v22 = v69;
      v48 = 0;
      v49 = ((((unsigned int)(37 * v66) | ((unsigned __int64)(unsigned int)(37 * v63) << 32))
            - 1
            - ((unsigned __int64)(unsigned int)(37 * v66) << 32)) >> 22)
          ^ (((unsigned int)(37 * v66) | ((unsigned __int64)(unsigned int)(37 * v63) << 32))
           - 1
           - ((unsigned __int64)(unsigned int)(37 * v66) << 32));
      v50 = ((9 * (((v49 - 1 - (v49 << 13)) >> 8) ^ (v49 - 1 - (v49 << 13)))) >> 15)
          ^ (9 * (((v49 - 1 - (v49 << 13)) >> 8) ^ (v49 - 1 - (v49 << 13))));
      for ( j = v45 & (((v50 - 1 - (v50 << 27)) >> 31) ^ (v50 - 1 - ((_DWORD)v50 << 27))); ; j = v45 & v53 )
      {
        v29 = v47 + 16LL * j;
        v52 = *(_DWORD *)v29;
        if ( v63 == *(_DWORD *)v29 && v66 == *(_DWORD *)(v29 + 4) )
          break;
        if ( v52 == -1 )
        {
          if ( *(_DWORD *)(v29 + 4) == -1 )
          {
            if ( v48 )
              v29 = v48;
            v43 = *(_DWORD *)(a1 + 416) + 1;
            goto LABEL_40;
          }
        }
        else if ( v52 == -2 && *(_DWORD *)(v29 + 4) == -2 && !v48 )
        {
          v48 = v47 + 16LL * j;
        }
        v53 = v46 + j;
        ++v46;
      }
      goto LABEL_66;
    }
LABEL_77:
    ++*(_DWORD *)(a1 + 416);
    BUG();
  }
  if ( v23 - *(_DWORD *)(a1 + 420) - v43 <= v23 >> 3 )
  {
    v64 = v18;
    v67 = v24;
    v70 = v22;
    sub_1F1A4C0(a1 + 400, v23);
    v54 = *(_DWORD *)(a1 + 424);
    if ( v54 )
    {
      v55 = v54 - 1;
      v22 = v70;
      v24 = v67;
      v56 = 1;
      v18 = v64;
      v58 = v55 & v27;
      v59 = 0;
      while ( 1 )
      {
        v57 = *(_QWORD *)(a1 + 408);
        v29 = v57 + 16LL * v58;
        v60 = *(_DWORD *)v29;
        if ( v64 == *(_DWORD *)v29 && v67 == *(_DWORD *)(v29 + 4) )
          break;
        if ( v60 == -1 )
        {
          if ( *(_DWORD *)(v29 + 4) == -1 )
          {
            if ( v59 )
              v29 = v59;
            v43 = *(_DWORD *)(a1 + 416) + 1;
            goto LABEL_40;
          }
        }
        else if ( v60 == -2 && *(_DWORD *)(v29 + 4) == -2 && !v59 )
        {
          v59 = v57 + 16LL * v58;
        }
        v61 = v56 + v58;
        ++v56;
        v58 = v55 & v61;
      }
LABEL_66:
      v43 = *(_DWORD *)(a1 + 416) + 1;
      goto LABEL_40;
    }
    goto LABEL_77;
  }
LABEL_40:
  *(_DWORD *)(a1 + 416) = v43;
  if ( *(_DWORD *)v29 != -1 || *(_DWORD *)(v29 + 4) != -1 )
    --*(_DWORD *)(a1 + 420);
  v34 = v77 | v73;
  *(_DWORD *)v29 = v18;
  *(_DWORD *)(v29 + 4) = v24;
  *(_QWORD *)(v29 + 8) = v77 | v73;
  if ( v22 )
    goto LABEL_22;
  return v19;
}
