// Function: sub_1646BA0
// Address: 0x1646ba0
//
__int64 __fastcall sub_1646BA0(__int64 *a1, int a2)
{
  __int64 v4; // rbx
  unsigned int v5; // esi
  __int64 v6; // rdi
  __int64 v7; // r8
  unsigned int v8; // ecx
  __int64 **v9; // rax
  __int64 *v10; // rdx
  __int64 v11; // r15
  __int64 *v12; // r14
  unsigned int v14; // esi
  __int64 v15; // rdi
  int v16; // r10d
  __int64 v17; // rdx
  __int64 v18; // r14
  unsigned __int64 v19; // rcx
  unsigned __int64 v20; // rcx
  unsigned __int64 v21; // r15
  unsigned int i; // r8d
  __int64 v23; // rax
  __int64 *v24; // rcx
  unsigned int v25; // r8d
  int v26; // eax
  int v27; // eax
  __int64 v28; // rdi
  unsigned int v29; // ecx
  int v30; // edx
  __int64 **v31; // r14
  __int64 *v32; // rsi
  int v33; // ecx
  int v34; // ecx
  __int64 v35; // rdx
  int v36; // r8d
  __int64 v37; // rdi
  unsigned __int64 v38; // rsi
  unsigned __int64 v39; // rsi
  unsigned int k; // eax
  __int64 *v41; // rsi
  unsigned int v42; // eax
  int v43; // eax
  int v44; // edx
  int v45; // r11d
  int v46; // eax
  int v47; // eax
  int v48; // eax
  __int64 v49; // rsi
  __int64 **v50; // rdi
  unsigned int v51; // r15d
  int v52; // r8d
  __int64 *v53; // rcx
  int v54; // eax
  int v55; // edx
  __int64 v56; // rsi
  unsigned int v57; // eax
  int j; // r8d
  __int64 *v59; // rcx
  unsigned int v60; // eax
  __int64 **v61; // r10
  int v62; // r9d
  __int64 **v63; // r8

  v4 = *(_QWORD *)*a1;
  if ( a2 )
  {
    v14 = *(_DWORD *)(v4 + 2632);
    v15 = v4 + 2608;
    if ( v14 )
    {
      v16 = 1;
      v17 = *(_QWORD *)(v4 + 2616);
      v18 = 0;
      v19 = ((((unsigned int)(37 * a2) | ((unsigned __int64)(((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4)) << 32))
            - 1
            - ((unsigned __int64)(unsigned int)(37 * a2) << 32)) >> 22)
          ^ (((unsigned int)(37 * a2) | ((unsigned __int64)(((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4)) << 32))
           - 1
           - ((unsigned __int64)(unsigned int)(37 * a2) << 32));
      v20 = ((9 * (((v19 - 1 - (v19 << 13)) >> 8) ^ (v19 - 1 - (v19 << 13)))) >> 15)
          ^ (9 * (((v19 - 1 - (v19 << 13)) >> 8) ^ (v19 - 1 - (v19 << 13))));
      v21 = (v20 - 1 - (v20 << 27)) ^ ((v20 - 1 - (v20 << 27)) >> 31);
      for ( i = v21 & (v14 - 1); ; i = (v14 - 1) & v25 )
      {
        v23 = v17 + 24LL * i;
        v24 = *(__int64 **)v23;
        if ( *(__int64 **)v23 == a1 && a2 == *(_DWORD *)(v23 + 8) )
        {
          v11 = *(_QWORD *)(v23 + 16);
          v12 = (__int64 *)(v23 + 16);
          goto LABEL_6;
        }
        if ( v24 == (__int64 *)-8LL )
        {
          if ( *(_DWORD *)(v23 + 8) == -1 )
          {
            if ( !v18 )
              v18 = v17 + 24LL * i;
            v43 = *(_DWORD *)(v4 + 2624);
            ++*(_QWORD *)(v4 + 2608);
            v44 = v43 + 1;
            if ( 4 * (v43 + 1) < 3 * v14 )
            {
              if ( v14 - *(_DWORD *)(v4 + 2628) - v44 > v14 >> 3 )
                goto LABEL_41;
              sub_16468F0(v15, v14);
              v54 = *(_DWORD *)(v4 + 2632);
              if ( v54 )
              {
                v55 = v54 - 1;
                v37 = 0;
                v57 = (v54 - 1) & v21;
                for ( j = 1; ; ++j )
                {
                  v56 = *(_QWORD *)(v4 + 2616);
                  v18 = v56 + 24LL * v57;
                  v59 = *(__int64 **)v18;
                  if ( *(__int64 **)v18 == a1 && a2 == *(_DWORD *)(v18 + 8) )
                    break;
                  if ( v59 == (__int64 *)-8LL )
                  {
                    if ( *(_DWORD *)(v18 + 8) == -1 )
                      goto LABEL_82;
                  }
                  else if ( v59 == (__int64 *)-16LL && *(_DWORD *)(v18 + 8) == -2 && !v37 )
                  {
                    v37 = v56 + 24LL * v57;
                  }
                  v60 = j + v57;
                  v57 = v55 & v60;
                }
LABEL_68:
                v44 = *(_DWORD *)(v4 + 2624) + 1;
LABEL_41:
                *(_DWORD *)(v4 + 2624) = v44;
                if ( *(_QWORD *)v18 != -8 || *(_DWORD *)(v18 + 8) != -1 )
                  --*(_DWORD *)(v4 + 2628);
                *(_QWORD *)v18 = a1;
                v12 = (__int64 *)(v18 + 16);
                *((_DWORD *)v12 - 2) = a2;
                *v12 = 0;
LABEL_23:
                v11 = sub_145CBF0((__int64 *)(v4 + 2272), 32, 16);
                sub_1643F30(v11, a1, a2);
                *v12 = v11;
                return v11;
              }
LABEL_98:
              ++*(_DWORD *)(v4 + 2624);
              BUG();
            }
LABEL_27:
            sub_16468F0(v15, 2 * v14);
            v33 = *(_DWORD *)(v4 + 2632);
            if ( v33 )
            {
              v34 = v33 - 1;
              v36 = 1;
              v37 = 0;
              v38 = ((((unsigned int)(37 * a2)
                     | ((unsigned __int64)(((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4)) << 32))
                    - 1
                    - ((unsigned __int64)(unsigned int)(37 * a2) << 32)) >> 22)
                  ^ (((unsigned int)(37 * a2)
                    | ((unsigned __int64)(((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4)) << 32))
                   - 1
                   - ((unsigned __int64)(unsigned int)(37 * a2) << 32));
              v39 = ((9 * (((v38 - 1 - (v38 << 13)) >> 8) ^ (v38 - 1 - (v38 << 13)))) >> 15)
                  ^ (9 * (((v38 - 1 - (v38 << 13)) >> 8) ^ (v38 - 1 - (v38 << 13))));
              for ( k = v34 & (((v39 - 1 - (v39 << 27)) >> 31) ^ (v39 - 1 - ((_DWORD)v39 << 27))); ; k = v34 & v42 )
              {
                v35 = *(_QWORD *)(v4 + 2616);
                v18 = v35 + 24LL * k;
                v41 = *(__int64 **)v18;
                if ( *(__int64 **)v18 == a1 && a2 == *(_DWORD *)(v18 + 8) )
                  goto LABEL_68;
                if ( v41 == (__int64 *)-8LL )
                {
                  if ( *(_DWORD *)(v18 + 8) == -1 )
                  {
LABEL_82:
                    if ( v37 )
                      v18 = v37;
                    v44 = *(_DWORD *)(v4 + 2624) + 1;
                    goto LABEL_41;
                  }
                }
                else if ( v41 == (__int64 *)-16LL && *(_DWORD *)(v18 + 8) == -2 && !v37 )
                {
                  v37 = v35 + 24LL * k;
                }
                v42 = v36 + k;
                ++v36;
              }
            }
            goto LABEL_98;
          }
        }
        else if ( v24 == (__int64 *)-16LL && *(_DWORD *)(v23 + 8) == -2 && !v18 )
        {
          v18 = v17 + 24LL * i;
        }
        v25 = v16 + i;
        ++v16;
      }
    }
    ++*(_QWORD *)(v4 + 2608);
    goto LABEL_27;
  }
  v5 = *(_DWORD *)(v4 + 2600);
  v6 = v4 + 2576;
  if ( !v5 )
  {
    ++*(_QWORD *)(v4 + 2576);
    goto LABEL_18;
  }
  v7 = *(_QWORD *)(v4 + 2584);
  v8 = (v5 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
  v9 = (__int64 **)(v7 + 16LL * v8);
  v10 = *v9;
  if ( *v9 == a1 )
  {
    v11 = (__int64)v9[1];
    goto LABEL_5;
  }
  v45 = 1;
  v31 = 0;
  while ( 1 )
  {
    if ( v10 == (__int64 *)-8LL )
    {
      if ( !v31 )
        v31 = v9;
      v46 = *(_DWORD *)(v4 + 2592);
      ++*(_QWORD *)(v4 + 2576);
      v30 = v46 + 1;
      if ( 4 * (v46 + 1) < 3 * v5 )
      {
        if ( v5 - *(_DWORD *)(v4 + 2596) - v30 > v5 >> 3 )
        {
LABEL_20:
          *(_DWORD *)(v4 + 2592) = v30;
          if ( *v31 != (__int64 *)-8LL )
            --*(_DWORD *)(v4 + 2596);
          *v31 = a1;
          v12 = (__int64 *)(v31 + 1);
          *v12 = 0;
          goto LABEL_23;
        }
        sub_1646730(v6, v5);
        v47 = *(_DWORD *)(v4 + 2600);
        if ( v47 )
        {
          v48 = v47 - 1;
          v49 = *(_QWORD *)(v4 + 2584);
          v50 = 0;
          v51 = v48 & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
          v52 = 1;
          v30 = *(_DWORD *)(v4 + 2592) + 1;
          v31 = (__int64 **)(v49 + 16LL * v51);
          v53 = *v31;
          if ( *v31 != a1 )
          {
            while ( v53 != (__int64 *)-8LL )
            {
              if ( v53 == (__int64 *)-16LL && !v50 )
                v50 = v31;
              v51 = v48 & (v52 + v51);
              v31 = (__int64 **)(v49 + 16LL * v51);
              v53 = *v31;
              if ( *v31 == a1 )
                goto LABEL_20;
              ++v52;
            }
            if ( v50 )
              v31 = v50;
          }
          goto LABEL_20;
        }
LABEL_99:
        ++*(_DWORD *)(v4 + 2592);
        BUG();
      }
LABEL_18:
      sub_1646730(v6, 2 * v5);
      v26 = *(_DWORD *)(v4 + 2600);
      if ( v26 )
      {
        v27 = v26 - 1;
        v28 = *(_QWORD *)(v4 + 2584);
        v29 = v27 & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
        v30 = *(_DWORD *)(v4 + 2592) + 1;
        v31 = (__int64 **)(v28 + 16LL * v29);
        v32 = *v31;
        if ( *v31 != a1 )
        {
          v62 = 1;
          v63 = 0;
          while ( v32 != (__int64 *)-8LL )
          {
            if ( !v63 && v32 == (__int64 *)-16LL )
              v63 = v31;
            v29 = v27 & (v62 + v29);
            v31 = (__int64 **)(v28 + 16LL * v29);
            v32 = *v31;
            if ( *v31 == a1 )
              goto LABEL_20;
            ++v62;
          }
          if ( v63 )
            v31 = v63;
        }
        goto LABEL_20;
      }
      goto LABEL_99;
    }
    if ( v10 != (__int64 *)-16LL || v31 )
      v9 = v31;
    v8 = (v5 - 1) & (v45 + v8);
    v61 = (__int64 **)(v7 + 16LL * v8);
    v10 = *v61;
    if ( *v61 == a1 )
      break;
    ++v45;
    v31 = v9;
    v9 = (__int64 **)(v7 + 16LL * v8);
  }
  v11 = (__int64)v61[1];
  v9 = (__int64 **)(v7 + 16LL * v8);
LABEL_5:
  v12 = (__int64 *)(v9 + 1);
LABEL_6:
  if ( !v11 )
    goto LABEL_23;
  return v11;
}
