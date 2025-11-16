// Function: sub_123FE40
// Address: 0x123fe40
//
__int64 __fastcall sub_123FE40(__int64 a1, __int64 a2, __int64 a3, unsigned __int64 a4, unsigned __int64 a5)
{
  __int64 v5; // rbx
  __int64 *v7; // rax
  __int64 *v8; // r9
  unsigned __int64 v9; // r12
  int v11; // esi
  __int64 v12; // r8
  unsigned int v13; // edx
  __int64 v14; // rax
  __int64 v15; // r10
  int v16; // r14d
  unsigned int v17; // esi
  unsigned int v18; // eax
  __int64 v19; // rdi
  unsigned int v20; // edx
  unsigned int v21; // r8d
  __int64 *v22; // r13
  unsigned int v23; // r12d
  int v25; // r11d
  __int64 v26; // r8
  int v27; // ecx
  unsigned int v28; // edx
  __int64 v29; // rsi
  int v30; // r10d
  __int64 v31; // rax
  __int64 v32; // r8
  int v33; // ecx
  unsigned int v34; // edx
  __int64 v35; // rsi
  int v36; // r10d
  __int64 v37; // rax
  __int64 v38; // rdx
  __int64 v39; // rcx
  unsigned int v40; // r15d
  __int64 *v41; // r11
  __int64 v42; // rbx
  __int64 v43; // rsi
  __int64 *v44; // r10
  int v45; // r8d
  __int64 v46; // r9
  unsigned int v47; // ecx
  __int64 v48; // rax
  __int64 v49; // r12
  unsigned int v50; // r12d
  unsigned int v51; // ecx
  __int64 v52; // rax
  __int64 v53; // rdi
  int v54; // eax
  int v55; // eax
  __int64 j; // rax
  __int64 v57; // rbx
  __int64 i; // r10
  __int64 *v59; // r9
  int v60; // edi
  __int64 v61; // r8
  unsigned int v62; // edx
  __int64 v63; // rax
  __int64 v64; // r12
  unsigned int v65; // r12d
  unsigned int v66; // edx
  __int64 v67; // rax
  __int64 v68; // rsi
  int v69; // eax
  int v70; // eax
  __int64 *v71; // [rsp+8h] [rbp-278h]
  __int64 *v72; // [rsp+8h] [rbp-278h]
  int v73; // [rsp+10h] [rbp-270h]
  int v74; // [rsp+10h] [rbp-270h]
  __int64 v78; // [rsp+28h] [rbp-258h]
  int v79; // [rsp+28h] [rbp-258h]
  int v80; // [rsp+28h] [rbp-258h]
  __int64 v81; // [rsp+38h] [rbp-248h] BYREF
  _QWORD v82[2]; // [rsp+40h] [rbp-240h] BYREF
  int v83; // [rsp+50h] [rbp-230h]
  __int16 v84; // [rsp+60h] [rbp-220h]
  const char *v85; // [rsp+140h] [rbp-140h] BYREF
  __int64 v86; // [rsp+148h] [rbp-138h]
  __int64 v87; // [rsp+150h] [rbp-130h] BYREF
  unsigned int v88; // [rsp+158h] [rbp-128h]
  char v89; // [rsp+160h] [rbp-120h]
  char v90; // [rsp+161h] [rbp-11Fh]
  char v91; // [rsp+250h] [rbp-30h] BYREF

  v5 = *(_QWORD *)(a2 + 16);
  if ( !v5 )
  {
    v90 = 1;
    v23 = 1;
    v85 = "value has no uses";
    v89 = 3;
    sub_11FD800(a1 + 176, a5, (__int64)&v85, 1);
    return v23;
  }
  v85 = 0;
  v86 = 1;
  v7 = &v87;
  do
  {
    *v7 = -4096;
    v7 += 2;
  }
  while ( v7 != (__int64 *)&v91 );
  v8 = &v87;
  v9 = 1;
  if ( !a4 )
  {
LABEL_93:
    v82[0] = "value only has one use";
    v84 = 259;
LABEL_24:
    v23 = 1;
    sub_11FD800(a1 + 176, a5, (__int64)v82, 1);
    goto LABEL_25;
  }
  do
  {
    v16 = *(_DWORD *)(a3 + 4LL * (unsigned int)(v9 - 1));
    if ( (v86 & 1) != 0 )
    {
      v11 = 15;
      v12 = (__int64)v8;
    }
    else
    {
      v17 = v88;
      v12 = v87;
      if ( !v88 )
      {
        v18 = v86;
        ++v85;
        v19 = 0;
        v20 = ((unsigned int)v86 >> 1) + 1;
LABEL_14:
        v21 = 3 * v17;
        goto LABEL_15;
      }
      v11 = v88 - 1;
    }
    v13 = v11 & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
    v14 = v12 + 16LL * v13;
    v15 = *(_QWORD *)v14;
    if ( v5 == *(_QWORD *)v14 )
    {
LABEL_8:
      *(_DWORD *)(v14 + 8) = v16;
      v5 = *(_QWORD *)(v5 + 8);
      if ( !v5 )
        break;
      goto LABEL_9;
    }
    v25 = 1;
    v19 = 0;
    while ( v15 != -4096 )
    {
      if ( !v19 && v15 == -8192 )
        v19 = v14;
      v13 = v11 & (v25 + v13);
      v14 = v12 + 16LL * v13;
      v15 = *(_QWORD *)v14;
      if ( *(_QWORD *)v14 == v5 )
        goto LABEL_8;
      ++v25;
    }
    v21 = 48;
    v17 = 16;
    if ( !v19 )
      v19 = v14;
    v18 = v86;
    ++v85;
    v20 = ((unsigned int)v86 >> 1) + 1;
    if ( (v86 & 1) == 0 )
    {
      v17 = v88;
      goto LABEL_14;
    }
LABEL_15:
    if ( 4 * v20 >= v21 )
    {
      v71 = v8;
      sub_9DEFB0((__int64)&v85, 2 * v17);
      v8 = v71;
      if ( (v86 & 1) != 0 )
      {
        v27 = 15;
        v26 = (__int64)v71;
      }
      else
      {
        v26 = v87;
        if ( !v88 )
          goto LABEL_139;
        v27 = v88 - 1;
      }
      v18 = v86;
      v28 = v27 & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
      v19 = v26 + 16LL * v28;
      v29 = *(_QWORD *)v19;
      if ( *(_QWORD *)v19 != v5 )
      {
        v30 = 1;
        v31 = 0;
        while ( v29 != -4096 )
        {
          if ( !v31 && v29 == -8192 )
            v31 = v19;
          v28 = v27 & (v30 + v28);
          v19 = v26 + 16LL * v28;
          v29 = *(_QWORD *)v19;
          if ( *(_QWORD *)v19 == v5 )
            goto LABEL_43;
          ++v30;
        }
        goto LABEL_41;
      }
    }
    else if ( v17 - HIDWORD(v86) - v20 <= v17 >> 3 )
    {
      v72 = v8;
      sub_9DEFB0((__int64)&v85, v17);
      v8 = v72;
      if ( (v86 & 1) != 0 )
      {
        v33 = 15;
        v32 = (__int64)v72;
      }
      else
      {
        v32 = v87;
        if ( !v88 )
        {
LABEL_139:
          LODWORD(v86) = (2 * ((unsigned int)v86 >> 1) + 2) | v86 & 1;
          BUG();
        }
        v33 = v88 - 1;
      }
      v18 = v86;
      v34 = v33 & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
      v19 = v32 + 16LL * v34;
      v35 = *(_QWORD *)v19;
      if ( v5 != *(_QWORD *)v19 )
      {
        v36 = 1;
        v31 = 0;
        while ( v35 != -4096 )
        {
          if ( v35 == -8192 && !v31 )
            v31 = v19;
          v34 = v33 & (v36 + v34);
          v19 = v32 + 16LL * v34;
          v35 = *(_QWORD *)v19;
          if ( *(_QWORD *)v19 == v5 )
            goto LABEL_43;
          ++v36;
        }
LABEL_41:
        if ( v31 )
          v19 = v31;
LABEL_43:
        v18 = v86;
      }
    }
    LODWORD(v86) = (2 * (v18 >> 1) + 2) | v18 & 1;
    if ( *(_QWORD *)v19 != -4096 )
      --HIDWORD(v86);
    *(_QWORD *)v19 = v5;
    *(_DWORD *)(v19 + 8) = 0;
    *(_DWORD *)(v19 + 8) = v16;
    v5 = *(_QWORD *)(v5 + 8);
    if ( !v5 )
      break;
LABEL_9:
    v9 = (unsigned int)(v9 + 1);
  }
  while ( v9 <= a4 );
  v22 = v8;
  if ( (unsigned int)v9 <= 1 )
    goto LABEL_93;
  if ( a4 != (unsigned int)v86 >> 1 || a4 < v9 )
  {
    v82[0] = "wrong number of indexes, expected ";
    v83 = sub_BD3960(a2);
    v84 = 2307;
    goto LABEL_24;
  }
  v37 = *(_QWORD *)(a2 + 16);
  if ( v37 )
  {
    v38 = *(_QWORD *)(v37 + 8);
    if ( v38 )
    {
      *(_QWORD *)(v37 + 8) = 0;
      v39 = *(_QWORD *)(a2 + 16);
      v82[0] = v39;
      if ( *(_QWORD *)(v38 + 8) )
      {
        v78 = *(_QWORD *)(v38 + 8);
        v40 = 1;
        while ( 1 )
        {
          *(_QWORD *)(v38 + 8) = 0;
          v41 = v82;
          v42 = 0;
          while ( 1 )
          {
            v43 = *v41;
            if ( !*v41 )
              break;
            v44 = &v81;
            while ( 1 )
            {
              if ( !v38 )
              {
LABEL_73:
                *v44 = v43;
                goto LABEL_74;
              }
              while ( 1 )
              {
                if ( (v86 & 1) != 0 )
                {
                  v45 = 15;
                  v46 = (__int64)v22;
                }
                else
                {
                  v46 = v87;
                  if ( !v88 )
                    break;
                  v45 = v88 - 1;
                }
                v47 = v45 & (((unsigned int)v38 >> 9) ^ ((unsigned int)v38 >> 4));
                v48 = v46 + 16LL * v47;
                v49 = *(_QWORD *)v48;
                if ( *(_QWORD *)v48 == v38 )
                {
LABEL_68:
                  v50 = *(_DWORD *)(v48 + 8);
                }
                else
                {
                  v54 = 1;
                  while ( v49 != -4096 )
                  {
                    v47 = v45 & (v54 + v47);
                    v74 = v54 + 1;
                    v48 = v46 + 16LL * v47;
                    v49 = *(_QWORD *)v48;
                    if ( *(_QWORD *)v48 == v38 )
                      goto LABEL_68;
                    v54 = v74;
                  }
                  v50 = 0;
                }
                if ( (v86 & 1) == 0 && !v88 )
                  break;
                v51 = v45 & (((unsigned int)v43 >> 9) ^ ((unsigned int)v43 >> 4));
                v52 = v46 + 16LL * v51;
                v53 = *(_QWORD *)v52;
                if ( *(_QWORD *)v52 != v43 )
                {
                  v55 = 1;
                  while ( v53 != -4096 )
                  {
                    v51 = v45 & (v55 + v51);
                    v73 = v55 + 1;
                    v52 = v46 + 16LL * v51;
                    v53 = *(_QWORD *)v52;
                    if ( *(_QWORD *)v52 == v43 )
                      goto LABEL_71;
                    v55 = v73;
                  }
                  break;
                }
LABEL_71:
                if ( *(_DWORD *)(v52 + 8) <= v50 )
                  break;
                *v44 = v38;
                v44 = (__int64 *)(v38 + 8);
                v38 = *(_QWORD *)(v38 + 8);
                if ( !v38 )
                  goto LABEL_73;
              }
              *v44 = v43;
              v44 = (__int64 *)(v43 + 8);
              if ( !*(_QWORD *)(v43 + 8) )
                break;
              v43 = *(_QWORD *)(v43 + 8);
            }
            *(_QWORD *)(v43 + 8) = v38;
LABEL_74:
            v42 = (unsigned int)(v42 + 1);
            *v41 = 0;
            v38 = v81;
            ++v41;
            if ( (_DWORD)v42 == v40 )
            {
              v40 = v42 + 1;
              break;
            }
          }
          v82[v42] = v38;
          v38 = v78;
          if ( !*(_QWORD *)(v78 + 8) )
            break;
          v78 = *(_QWORD *)(v78 + 8);
        }
        v57 = v78;
        v39 = v82[0];
        *(_QWORD *)(a2 + 16) = v78;
      }
      else
      {
        v57 = v38;
        v40 = 1;
        *(_QWORD *)(a2 + 16) = v38;
      }
      for ( i = 0; ; v39 = v82[i] )
      {
        v59 = &v81;
        if ( v39 )
        {
          while ( 1 )
          {
            if ( !v57 )
            {
LABEL_109:
              *v59 = v39;
              goto LABEL_110;
            }
            while ( 1 )
            {
              if ( (v86 & 1) != 0 )
              {
                v60 = 15;
                v61 = (__int64)v22;
              }
              else
              {
                v61 = v87;
                if ( !v88 )
                  break;
                v60 = v88 - 1;
              }
              v62 = v60 & (((unsigned int)v57 >> 9) ^ ((unsigned int)v57 >> 4));
              v63 = v61 + 16LL * v62;
              v64 = *(_QWORD *)v63;
              if ( *(_QWORD *)v63 == v57 )
              {
LABEL_104:
                v65 = *(_DWORD *)(v63 + 8);
              }
              else
              {
                v70 = 1;
                while ( v64 != -4096 )
                {
                  v62 = v60 & (v70 + v62);
                  v80 = v70 + 1;
                  v63 = v61 + 16LL * v62;
                  v64 = *(_QWORD *)v63;
                  if ( *(_QWORD *)v63 == v57 )
                    goto LABEL_104;
                  v70 = v80;
                }
                v65 = 0;
              }
              if ( (v86 & 1) == 0 && !v88 )
                break;
              v66 = v60 & (((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4));
              v67 = v61 + 16LL * v66;
              v68 = *(_QWORD *)v67;
              if ( *(_QWORD *)v67 != v39 )
              {
                v69 = 1;
                while ( v68 != -4096 )
                {
                  v66 = v60 & (v69 + v66);
                  v79 = v69 + 1;
                  v67 = v61 + 16LL * v66;
                  v68 = *(_QWORD *)v67;
                  if ( *(_QWORD *)v67 == v39 )
                    goto LABEL_107;
                  v69 = v79;
                }
                break;
              }
LABEL_107:
              if ( *(_DWORD *)(v67 + 8) <= v65 )
                break;
              *v59 = v57;
              v59 = (__int64 *)(v57 + 8);
              v57 = *(_QWORD *)(v57 + 8);
              if ( !v57 )
                goto LABEL_109;
            }
            *v59 = v39;
            v59 = (__int64 *)(v39 + 8);
            if ( !*(_QWORD *)(v39 + 8) )
              break;
            v39 = *(_QWORD *)(v39 + 8);
          }
          *(_QWORD *)(v39 + 8) = v57;
LABEL_110:
          v57 = v81;
          *(_QWORD *)(a2 + 16) = v81;
        }
        if ( v40 <= (unsigned int)++i )
          break;
      }
      for ( j = a2 + 16; v57; v57 = *(_QWORD *)(v57 + 8) )
      {
        *(_QWORD *)(v57 + 16) = j;
        j = v57 + 8;
      }
    }
  }
  v23 = 0;
LABEL_25:
  if ( (v86 & 1) == 0 )
    sub_C7D6A0(v87, 16LL * v88, 8);
  return v23;
}
