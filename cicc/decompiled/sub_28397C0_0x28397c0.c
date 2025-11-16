// Function: sub_28397C0
// Address: 0x28397c0
//
__int64 __fastcall sub_28397C0(__int64 a1, _QWORD *a2)
{
  _QWORD *v2; // rbx
  __int64 v4; // rsi
  __int64 v5; // rdi
  int v6; // r10d
  _QWORD *v7; // rax
  unsigned int v8; // ecx
  __int64 v9; // r12
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rsi
  __int64 v13; // r14
  _QWORD *v14; // r13
  unsigned int v15; // ecx
  int v16; // edx
  __int64 v17; // r9
  int v18; // r11d
  _QWORD *v19; // r10
  _QWORD **v20; // r13
  _QWORD *v21; // rbx
  __int64 v22; // rcx
  _QWORD *v23; // r10
  int v24; // r14d
  unsigned int v25; // edx
  _QWORD *v26; // rax
  __int64 v27; // r11
  _QWORD *v28; // rdi
  __int64 v29; // rsi
  unsigned int v30; // edx
  int v31; // eax
  __int64 v32; // r8
  __int64 v33; // rax
  __int64 v35; // rsi
  _QWORD *v36; // r9
  int v37; // r11d
  unsigned int v38; // edx
  __int64 v39; // r8
  __int64 v40; // rsi
  unsigned int v41; // ecx
  __int64 v42; // rdi
  unsigned int v43; // r9d
  unsigned int v44; // edx
  __int64 *v45; // rax
  __int64 v46; // r11
  unsigned int v47; // r10d
  __int64 v48; // rsi
  unsigned int v49; // edx
  __int64 *v50; // rax
  __int64 v51; // r14
  unsigned int v52; // eax
  int v53; // r10d
  unsigned int v54; // r15d
  _QWORD *v55; // r9
  __int64 v56; // rsi
  int v57; // eax
  int v58; // r11d
  int v59; // eax
  int v60; // r8d
  int v61; // r8d
  __int64 v62; // rax
  _QWORD *v63; // [rsp+10h] [rbp-60h]
  __int64 v65; // [rsp+20h] [rbp-50h] BYREF
  __int64 v66; // [rsp+28h] [rbp-48h]
  __int64 v67; // [rsp+30h] [rbp-40h]
  unsigned int v68; // [rsp+38h] [rbp-38h]

  v2 = (_QWORD *)*a2;
  v65 = 0;
  v66 = 0;
  v67 = 0;
  v68 = 0;
  if ( !v2 )
  {
    v5 = 0;
    v4 = 0;
    return sub_C7D6A0(v5, 16 * v4, 8);
  }
  v4 = 0;
  v5 = 0;
  v63 = a2;
  do
  {
    v13 = v2[1];
    v14 = v2 + 1;
    if ( !(_DWORD)v4 )
    {
      ++v65;
      goto LABEL_11;
    }
    v6 = 1;
    v7 = 0;
    v8 = (v4 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
    v9 = v5 + 16LL * v8;
    v10 = *(_QWORD *)v9;
    if ( v13 != *(_QWORD *)v9 )
    {
      while ( v10 != -4096 )
      {
        if ( v10 == -8192 && !v7 )
          v7 = (_QWORD *)v9;
        v8 = (v4 - 1) & (v6 + v8);
        v9 = v5 + 16LL * v8;
        v10 = *(_QWORD *)v9;
        if ( v13 == *(_QWORD *)v9 )
          goto LABEL_4;
        ++v6;
      }
      if ( !v7 )
        v7 = (_QWORD *)v9;
      ++v65;
      v16 = v67 + 1;
      if ( 4 * ((int)v67 + 1) < (unsigned int)(3 * v4) )
      {
        if ( (int)v4 - (v16 + HIDWORD(v67)) <= (unsigned int)v4 >> 3 )
        {
          sub_2839380((__int64)&v65, v4);
          if ( !v68 )
          {
LABEL_110:
            LODWORD(v67) = v67 + 1;
            BUG();
          }
          v53 = 1;
          v54 = (v68 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
          v55 = 0;
          v16 = v67 + 1;
          v7 = (_QWORD *)(v66 + 16LL * v54);
          v56 = *v7;
          if ( v13 != *v7 )
          {
            while ( v56 != -4096 )
            {
              if ( v56 == -8192 && !v55 )
                v55 = v7;
              v54 = (v68 - 1) & (v53 + v54);
              v7 = (_QWORD *)(v66 + 16LL * v54);
              v56 = *v7;
              if ( v13 == *v7 )
                goto LABEL_28;
              ++v53;
            }
            if ( v55 )
              v7 = v55;
          }
        }
        goto LABEL_28;
      }
LABEL_11:
      sub_2839380((__int64)&v65, 2 * v4);
      if ( !v68 )
        goto LABEL_110;
      v15 = (v68 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
      v16 = v67 + 1;
      v7 = (_QWORD *)(v66 + 16LL * v15);
      v17 = *v7;
      if ( v13 != *v7 )
      {
        v18 = 1;
        v19 = 0;
        while ( v17 != -4096 )
        {
          if ( !v19 && v17 == -8192 )
            v19 = v7;
          v15 = (v68 - 1) & (v18 + v15);
          v7 = (_QWORD *)(v66 + 16LL * v15);
          v17 = *v7;
          if ( v13 == *v7 )
            goto LABEL_28;
          ++v18;
        }
        if ( v19 )
          v7 = v19;
      }
LABEL_28:
      LODWORD(v67) = v16;
      if ( *v7 != -4096 )
        --HIDWORD(v67);
      *v7 = v13;
      v7[1] = v14;
      goto LABEL_7;
    }
LABEL_4:
    v11 = *(_QWORD *)(v9 + 8);
    if ( !v11 )
      goto LABEL_8;
    v12 = v2[2];
    if ( *(_QWORD *)(*(_QWORD *)(v11 + 8) + 40LL) == *(_QWORD *)(v12 + 40)
      && (unsigned __int8)sub_2839560(v2[1], v12, a1 + 80, *(_QWORD *)a1)
      && (unsigned __int8)sub_2839560(
                            **(_QWORD **)(v9 + 8),
                            *(_QWORD *)(*(_QWORD *)(v9 + 8) + 8LL),
                            a1 + 80,
                            *(_QWORD *)a1) )
    {
      v40 = *(_QWORD *)(*(_QWORD *)(v9 + 8) + 8LL);
      v41 = *(_DWORD *)(a1 + 32);
      v42 = *(_QWORD *)(a1 + 16);
      if ( v41 )
      {
        v43 = v41 - 1;
        v44 = (v41 - 1) & (((unsigned int)v40 >> 9) ^ ((unsigned int)v40 >> 4));
        v45 = (__int64 *)(v42 + 16LL * v44);
        v46 = *v45;
        if ( v40 == *v45 )
        {
LABEL_65:
          v47 = *((_DWORD *)v45 + 2);
          v48 = v2[2];
        }
        else
        {
          v57 = 1;
          while ( v46 != -4096 )
          {
            v60 = v57 + 1;
            v44 = v43 & (v57 + v44);
            v45 = (__int64 *)(v42 + 16LL * v44);
            v46 = *v45;
            if ( v40 == *v45 )
              goto LABEL_65;
            v57 = v60;
          }
          v48 = v2[2];
          v47 = *(_DWORD *)(v42 + 16LL * v41 + 8);
        }
        v49 = v43 & (((unsigned int)v48 >> 9) ^ ((unsigned int)v48 >> 4));
        v50 = (__int64 *)(v42 + 16LL * v49);
        v51 = *v50;
        if ( *v50 == v48 )
        {
LABEL_67:
          v52 = *((_DWORD *)v50 + 2);
        }
        else
        {
          v59 = 1;
          while ( v51 != -4096 )
          {
            v61 = v59 + 1;
            v62 = v43 & (v49 + v59);
            v49 = v62;
            v50 = (__int64 *)(v42 + 16 * v62);
            v51 = *v50;
            if ( *v50 == v48 )
              goto LABEL_67;
            v59 = v61;
          }
          v52 = *(_DWORD *)(v42 + 16LL * v41 + 8);
        }
        if ( v52 > v47 )
          *(_QWORD *)(v9 + 8) = v14;
      }
    }
    else
    {
      *(_QWORD *)(v9 + 8) = 0;
    }
LABEL_7:
    v5 = v66;
    v4 = v68;
LABEL_8:
    v2 = (_QWORD *)*v2;
  }
  while ( v2 );
  v20 = (_QWORD **)v63;
  v21 = (_QWORD *)*v63;
  if ( *v63 )
  {
    while ( (_DWORD)v4 )
    {
      v22 = v21[1];
      v23 = 0;
      v24 = 1;
      v25 = (v4 - 1) & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
      v26 = (_QWORD *)(v5 + 16LL * v25);
      v27 = *v26;
      if ( *v26 != v22 )
      {
        while ( v27 != -4096 )
        {
          if ( v27 == -8192 && !v23 )
            v23 = v26;
          v25 = (v4 - 1) & (v24 + v25);
          v26 = (_QWORD *)(v5 + 16LL * v25);
          v27 = *v26;
          if ( v22 == *v26 )
            goto LABEL_34;
          ++v24;
        }
        if ( !v23 )
          v23 = v26;
        ++v65;
        v31 = v67 + 1;
        if ( 4 * ((int)v67 + 1) < (unsigned int)(3 * v4) )
        {
          if ( (int)v4 - (v31 + HIDWORD(v67)) <= (unsigned int)v4 >> 3 )
          {
            sub_2839380((__int64)&v65, v4);
            if ( !v68 )
            {
LABEL_109:
              LODWORD(v67) = v67 + 1;
              BUG();
            }
            v35 = v21[1];
            v36 = 0;
            v37 = 1;
            v38 = (v68 - 1) & (((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4));
            v31 = v67 + 1;
            v23 = (_QWORD *)(v66 + 16LL * v38);
            v39 = *v23;
            if ( v35 != *v23 )
            {
              while ( v39 != -4096 )
              {
                if ( v39 == -8192 && !v36 )
                  v36 = v23;
                v38 = (v68 - 1) & (v37 + v38);
                v23 = (_QWORD *)(v66 + 16LL * v38);
                v39 = *v23;
                if ( v35 == *v23 )
                  goto LABEL_40;
                ++v37;
              }
LABEL_58:
              if ( v36 )
                v23 = v36;
            }
          }
LABEL_40:
          LODWORD(v67) = v31;
          if ( *v23 != -4096 )
            --HIDWORD(v67);
          v33 = v21[1];
          v23[1] = 0;
          *v23 = v33;
          v28 = *v20;
          goto LABEL_43;
        }
LABEL_38:
        sub_2839380((__int64)&v65, 2 * v4);
        if ( !v68 )
          goto LABEL_109;
        v29 = v21[1];
        v30 = (v68 - 1) & (((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4));
        v31 = v67 + 1;
        v23 = (_QWORD *)(v66 + 16LL * v30);
        v32 = *v23;
        if ( *v23 != v29 )
        {
          v36 = 0;
          v58 = 1;
          while ( v32 != -4096 )
          {
            if ( !v36 && v32 == -8192 )
              v36 = v23;
            v30 = (v68 - 1) & (v58 + v30);
            v23 = (_QWORD *)(v66 + 16LL * v30);
            v32 = *v23;
            if ( v29 == *v23 )
              goto LABEL_40;
            ++v58;
          }
          goto LABEL_58;
        }
        goto LABEL_40;
      }
LABEL_34:
      v28 = *v20;
      if ( (_QWORD *)v26[1] == v21 + 1 )
      {
        v20 = (_QWORD **)*v20;
        v4 = v68;
        v5 = v66;
        v21 = *v20;
        if ( !*v20 )
          return sub_C7D6A0(v5, 16 * v4, 8);
      }
      else
      {
LABEL_43:
        *v20 = (_QWORD *)*v28;
        j_j___libc_free_0((unsigned __int64)v28);
        v21 = *v20;
        v4 = v68;
        v5 = v66;
        if ( !*v20 )
          return sub_C7D6A0(v5, 16 * v4, 8);
      }
    }
    ++v65;
    goto LABEL_38;
  }
  return sub_C7D6A0(v5, 16 * v4, 8);
}
