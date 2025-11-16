// Function: sub_38B6F20
// Address: 0x38b6f20
//
__int64 __fastcall sub_38B6F20(__int64 a1, __int64 a2, __int64 a3, unsigned __int64 a4, unsigned __int64 a5)
{
  __int64 v5; // rbx
  unsigned __int64 *v7; // rax
  unsigned __int64 *v8; // r9
  unsigned __int64 v9; // r12
  int v11; // esi
  unsigned __int64 v12; // rdi
  unsigned int v13; // edx
  unsigned __int64 v14; // rax
  __int64 v15; // r8
  _DWORD *v16; // r14
  unsigned int v17; // esi
  unsigned int v18; // edx
  unsigned int v19; // edi
  unsigned int v20; // r8d
  unsigned __int64 *v21; // r13
  __int64 result; // rax
  int v23; // r11d
  unsigned __int64 v24; // r10
  unsigned __int64 v25; // r8
  int v26; // esi
  unsigned int v27; // ecx
  __int64 v28; // rdi
  int v29; // r10d
  unsigned __int64 v30; // rdx
  unsigned __int64 v31; // r8
  int v32; // esi
  unsigned int v33; // ecx
  __int64 v34; // rdi
  int v35; // r10d
  __int64 v36; // rdx
  __int64 v37; // rax
  __int64 v38; // rdx
  unsigned int v39; // r15d
  __int64 *v40; // r11
  __int64 v41; // rbx
  __int64 v42; // rdi
  _QWORD *v43; // r10
  int v44; // r8d
  unsigned __int64 v45; // r9
  unsigned int v46; // esi
  unsigned __int64 v47; // rdx
  __int64 v48; // r12
  unsigned int v49; // esi
  unsigned int v50; // ecx
  unsigned __int64 v51; // rdx
  __int64 v52; // r12
  int v53; // edx
  int v54; // edx
  __int64 j; // rax
  __int64 v56; // rbx
  __int64 i; // r10
  _QWORD *v58; // r9
  int v59; // edi
  unsigned __int64 v60; // r8
  unsigned int v61; // esi
  unsigned __int64 v62; // rax
  __int64 v63; // r12
  unsigned int v64; // esi
  unsigned int v65; // ecx
  unsigned __int64 v66; // rax
  __int64 v67; // r12
  int v68; // eax
  int v69; // eax
  unsigned __int64 *v70; // [rsp+8h] [rbp-288h]
  unsigned __int64 *v71; // [rsp+8h] [rbp-288h]
  int v72; // [rsp+10h] [rbp-280h]
  int v73; // [rsp+10h] [rbp-280h]
  unsigned __int8 v77; // [rsp+28h] [rbp-268h]
  __int64 v78; // [rsp+28h] [rbp-268h]
  int v79; // [rsp+28h] [rbp-268h]
  int v80; // [rsp+28h] [rbp-268h]
  _QWORD v81[4]; // [rsp+30h] [rbp-260h] BYREF
  _QWORD v82[2]; // [rsp+50h] [rbp-240h] BYREF
  __int16 v83; // [rsp+60h] [rbp-230h]
  const char *v84; // [rsp+150h] [rbp-140h] BYREF
  __int64 v85; // [rsp+158h] [rbp-138h]
  unsigned __int64 v86; // [rsp+160h] [rbp-130h] BYREF
  unsigned int v87; // [rsp+168h] [rbp-128h]
  char v88; // [rsp+260h] [rbp-30h] BYREF

  v5 = *(_QWORD *)(a2 + 8);
  if ( !v5 )
  {
    v84 = "value has no uses";
    LOWORD(v86) = 259;
    return sub_38814C0(a1 + 8, a5, (__int64)&v84);
  }
  v84 = 0;
  v85 = 1;
  v7 = &v86;
  do
  {
    *v7 = -8;
    v7 += 2;
  }
  while ( v7 != (unsigned __int64 *)&v88 );
  v8 = &v86;
  v9 = 1;
  if ( !a4 )
  {
LABEL_93:
    v82[0] = "value only has one use";
    v83 = 259;
LABEL_24:
    result = sub_38814C0(a1 + 8, a5, (__int64)v82);
    goto LABEL_25;
  }
  do
  {
    v16 = (_DWORD *)(a3 + 4LL * (unsigned int)(v9 - 1));
    if ( (v85 & 1) != 0 )
    {
      v11 = 15;
      v12 = (unsigned __int64)v8;
    }
    else
    {
      v17 = v87;
      v12 = v86;
      if ( !v87 )
      {
        v18 = v85;
        ++v84;
        v14 = 0;
        v19 = ((unsigned int)v85 >> 1) + 1;
LABEL_14:
        v20 = 3 * v17;
        goto LABEL_15;
      }
      v11 = v87 - 1;
    }
    v13 = v11 & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
    v14 = v12 + 16LL * v13;
    v15 = *(_QWORD *)v14;
    if ( v5 == *(_QWORD *)v14 )
    {
LABEL_8:
      *(_DWORD *)(v14 + 8) = *v16;
      v5 = *(_QWORD *)(v5 + 8);
      if ( !v5 )
        break;
      goto LABEL_9;
    }
    v23 = 1;
    v24 = 0;
    while ( v15 != -8 )
    {
      if ( v15 == -16 && !v24 )
        v24 = v14;
      v13 = v11 & (v23 + v13);
      v14 = v12 + 16LL * v13;
      v15 = *(_QWORD *)v14;
      if ( *(_QWORD *)v14 == v5 )
        goto LABEL_8;
      ++v23;
    }
    v18 = v85;
    v20 = 48;
    v17 = 16;
    if ( v24 )
      v14 = v24;
    ++v84;
    v19 = ((unsigned int)v85 >> 1) + 1;
    if ( (v85 & 1) == 0 )
    {
      v17 = v87;
      goto LABEL_14;
    }
LABEL_15:
    if ( 4 * v19 >= v20 )
    {
      v70 = v8;
      sub_14FBFD0((__int64)&v84, 2 * v17);
      v8 = v70;
      if ( (v85 & 1) != 0 )
      {
        v26 = 15;
        v25 = (unsigned __int64)v70;
      }
      else
      {
        v25 = v86;
        if ( !v87 )
          goto LABEL_140;
        v26 = v87 - 1;
      }
      v18 = v85;
      v27 = v26 & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
      v14 = v25 + 16LL * v27;
      v28 = *(_QWORD *)v14;
      if ( *(_QWORD *)v14 != v5 )
      {
        v29 = 1;
        v30 = 0;
        while ( v28 != -8 )
        {
          if ( v28 == -16 && !v30 )
            v30 = v14;
          v27 = v26 & (v29 + v27);
          v14 = v25 + 16LL * v27;
          v28 = *(_QWORD *)v14;
          if ( v5 == *(_QWORD *)v14 )
            goto LABEL_43;
          ++v29;
        }
        goto LABEL_41;
      }
    }
    else if ( v17 - HIDWORD(v85) - v19 <= v17 >> 3 )
    {
      v71 = v8;
      sub_14FBFD0((__int64)&v84, v17);
      v8 = v71;
      if ( (v85 & 1) != 0 )
      {
        v32 = 15;
        v31 = (unsigned __int64)v71;
      }
      else
      {
        v31 = v86;
        if ( !v87 )
        {
LABEL_140:
          LODWORD(v85) = (2 * ((unsigned int)v85 >> 1) + 2) | v85 & 1;
          BUG();
        }
        v32 = v87 - 1;
      }
      v18 = v85;
      v33 = v32 & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
      v14 = v31 + 16LL * v33;
      v34 = *(_QWORD *)v14;
      if ( v5 != *(_QWORD *)v14 )
      {
        v35 = 1;
        v30 = 0;
        while ( v34 != -8 )
        {
          if ( v34 == -16 && !v30 )
            v30 = v14;
          v33 = v32 & (v35 + v33);
          v14 = v31 + 16LL * v33;
          v34 = *(_QWORD *)v14;
          if ( v5 == *(_QWORD *)v14 )
            goto LABEL_43;
          ++v35;
        }
LABEL_41:
        if ( v30 )
          v14 = v30;
LABEL_43:
        v18 = v85;
      }
    }
    LODWORD(v85) = (2 * (v18 >> 1) + 2) | v18 & 1;
    if ( *(_QWORD *)v14 != -8 )
      --HIDWORD(v85);
    *(_DWORD *)(v14 + 8) = 0;
    *(_QWORD *)v14 = v5;
    *(_DWORD *)(v14 + 8) = *v16;
    v5 = *(_QWORD *)(v5 + 8);
    if ( !v5 )
      break;
LABEL_9:
    v9 = (unsigned int)(v9 + 1);
  }
  while ( v9 <= a4 );
  v21 = v8;
  if ( (unsigned int)v9 <= 1 )
    goto LABEL_93;
  if ( a4 != (unsigned int)v85 >> 1 || a4 < v9 )
  {
    LODWORD(v81[0]) = sub_1648EF0(a2);
    v82[0] = "wrong number of indexes, expected ";
    v83 = 2307;
    v82[1] = v81[0];
    goto LABEL_24;
  }
  v36 = *(_QWORD *)(a2 + 8);
  if ( v36 )
  {
    v37 = *(_QWORD *)(v36 + 8);
    if ( v37 )
    {
      *(_QWORD *)(v36 + 8) = 0;
      v38 = *(_QWORD *)(a2 + 8);
      v82[0] = v38;
      if ( *(_QWORD *)(v37 + 8) )
      {
        v78 = *(_QWORD *)(v37 + 8);
        v39 = 1;
        while ( 1 )
        {
          *(_QWORD *)(v37 + 8) = 0;
          v40 = v82;
          v41 = 0;
          while ( 1 )
          {
            v42 = *v40;
            if ( !*v40 )
              break;
            v43 = v81;
            while ( 1 )
            {
              if ( !v37 )
              {
LABEL_73:
                *v43 = v42;
                goto LABEL_74;
              }
              while ( 1 )
              {
                if ( (v85 & 1) != 0 )
                {
                  v44 = 15;
                  v45 = (unsigned __int64)v21;
                }
                else
                {
                  v45 = v86;
                  if ( !v87 )
                    break;
                  v44 = v87 - 1;
                }
                v46 = v44 & (((unsigned int)v37 >> 9) ^ ((unsigned int)v37 >> 4));
                v47 = v45 + 16LL * v46;
                v48 = *(_QWORD *)v47;
                if ( *(_QWORD *)v47 == v37 )
                {
LABEL_68:
                  v49 = *(_DWORD *)(v47 + 8);
                }
                else
                {
                  v53 = 1;
                  while ( v48 != -8 )
                  {
                    v46 = v44 & (v53 + v46);
                    v73 = v53 + 1;
                    v47 = v45 + 16LL * v46;
                    v48 = *(_QWORD *)v47;
                    if ( *(_QWORD *)v47 == v37 )
                      goto LABEL_68;
                    v53 = v73;
                  }
                  v49 = 0;
                }
                if ( (v85 & 1) == 0 && !v87 )
                  break;
                v50 = v44 & (((unsigned int)v42 >> 9) ^ ((unsigned int)v42 >> 4));
                v51 = v45 + 16LL * v50;
                v52 = *(_QWORD *)v51;
                if ( *(_QWORD *)v51 != v42 )
                {
                  v54 = 1;
                  while ( v52 != -8 )
                  {
                    v50 = v44 & (v54 + v50);
                    v72 = v54 + 1;
                    v51 = v45 + 16LL * v50;
                    v52 = *(_QWORD *)v51;
                    if ( *(_QWORD *)v51 == v42 )
                      goto LABEL_71;
                    v54 = v72;
                  }
                  break;
                }
LABEL_71:
                if ( *(_DWORD *)(v51 + 8) <= v49 )
                  break;
                *v43 = v37;
                v43 = (_QWORD *)(v37 + 8);
                v37 = *(_QWORD *)(v37 + 8);
                if ( !v37 )
                  goto LABEL_73;
              }
              *v43 = v42;
              v43 = (_QWORD *)(v42 + 8);
              if ( !*(_QWORD *)(v42 + 8) )
                break;
              v42 = *(_QWORD *)(v42 + 8);
            }
            *(_QWORD *)(v42 + 8) = v37;
LABEL_74:
            v41 = (unsigned int)(v41 + 1);
            *v40 = 0;
            v37 = v81[0];
            ++v40;
            if ( v39 == (_DWORD)v41 )
            {
              ++v39;
              break;
            }
          }
          v82[v41] = v37;
          v37 = v78;
          if ( !*(_QWORD *)(v78 + 8) )
            break;
          v78 = *(_QWORD *)(v78 + 8);
        }
        v56 = v78;
        v38 = v82[0];
        *(_QWORD *)(a2 + 8) = v78;
      }
      else
      {
        v39 = 1;
        *(_QWORD *)(a2 + 8) = v37;
        v56 = v37;
      }
      for ( i = 0; ; v38 = v82[i] )
      {
        v58 = v81;
        if ( v38 )
        {
          while ( 1 )
          {
            if ( !v56 )
            {
LABEL_109:
              *v58 = v38;
              goto LABEL_110;
            }
            while ( 1 )
            {
              if ( (v85 & 1) != 0 )
              {
                v59 = 15;
                v60 = (unsigned __int64)v21;
              }
              else
              {
                v60 = v86;
                if ( !v87 )
                  break;
                v59 = v87 - 1;
              }
              v61 = v59 & (((unsigned int)v56 >> 9) ^ ((unsigned int)v56 >> 4));
              v62 = v60 + 16LL * v61;
              v63 = *(_QWORD *)v62;
              if ( *(_QWORD *)v62 == v56 )
              {
LABEL_104:
                v64 = *(_DWORD *)(v62 + 8);
              }
              else
              {
                v69 = 1;
                while ( v63 != -8 )
                {
                  v61 = v59 & (v69 + v61);
                  v80 = v69 + 1;
                  v62 = v60 + 16LL * v61;
                  v63 = *(_QWORD *)v62;
                  if ( *(_QWORD *)v62 == v56 )
                    goto LABEL_104;
                  v69 = v80;
                }
                v64 = 0;
              }
              if ( (v85 & 1) == 0 && !v87 )
                break;
              v65 = v59 & (((unsigned int)v38 >> 9) ^ ((unsigned int)v38 >> 4));
              v66 = v60 + 16LL * v65;
              v67 = *(_QWORD *)v66;
              if ( *(_QWORD *)v66 != v38 )
              {
                v68 = 1;
                while ( v67 != -8 )
                {
                  v65 = v59 & (v68 + v65);
                  v79 = v68 + 1;
                  v66 = v60 + 16LL * v65;
                  v67 = *(_QWORD *)v66;
                  if ( *(_QWORD *)v66 == v38 )
                    goto LABEL_107;
                  v68 = v79;
                }
                break;
              }
LABEL_107:
              if ( *(_DWORD *)(v66 + 8) <= v64 )
                break;
              *v58 = v56;
              v58 = (_QWORD *)(v56 + 8);
              v56 = *(_QWORD *)(v56 + 8);
              if ( !v56 )
                goto LABEL_109;
            }
            *v58 = v38;
            v58 = (_QWORD *)(v38 + 8);
            if ( !*(_QWORD *)(v38 + 8) )
              break;
            v38 = *(_QWORD *)(v38 + 8);
          }
          *(_QWORD *)(v38 + 8) = v56;
LABEL_110:
          v56 = v81[0];
          *(_QWORD *)(a2 + 8) = v81[0];
        }
        if ( v39 <= (unsigned int)++i )
          break;
      }
      for ( j = a2 + 8; v56; v56 = *(_QWORD *)(v56 + 8) )
      {
        *(_QWORD *)(v56 + 16) = *(_QWORD *)(v56 + 16) & 3LL | j;
        j = v56 + 8;
      }
    }
  }
  result = 0;
LABEL_25:
  if ( (v85 & 1) == 0 )
  {
    v77 = result;
    j___libc_free_0(v86);
    return v77;
  }
  return result;
}
