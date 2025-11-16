// Function: sub_2A3C3D0
// Address: 0x2a3c3d0
//
void __fastcall sub_2A3C3D0(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 v4; // rdi
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rbx
  __int64 v8; // r15
  __int64 v9; // r8
  unsigned __int8 *v10; // r12
  unsigned __int64 *v11; // r14
  unsigned __int64 v12; // rdx
  __int64 v13; // rbx
  _BYTE *v14; // rsi
  char v15; // al
  __int64 v16; // rdx
  unsigned int v17; // eax
  unsigned __int8 *v18; // rbx
  __int64 v19; // r12
  unsigned __int8 *v20; // rax
  __int64 v21; // rax
  _QWORD *v22; // r12
  _QWORD *v23; // rdx
  __int64 v24; // r15
  _BYTE *v25; // rsi
  __int64 v26; // rax
  __int64 v27; // r8
  __int64 v28; // r9
  __int64 v29; // rdx
  unsigned __int64 v30; // r10
  unsigned __int8 *v31; // rax
  __int64 v32; // r8
  __int64 v33; // r9
  __int64 v34; // r12
  __int64 v35; // rax
  int v36; // eax
  __int64 v37; // rbx
  __int64 v38; // r12
  __int64 v39; // rax
  unsigned __int64 *v40; // rbx
  unsigned __int64 *v41; // r12
  unsigned __int64 v42; // rdi
  __int64 v43; // r12
  __int64 v44; // rax
  unsigned __int64 *v45; // rbx
  unsigned __int64 v46; // rdi
  unsigned __int64 v47; // rbx
  __int64 v48; // r8
  __int64 v49; // r9
  __int64 v50; // rax
  __int64 v51; // r8
  __int64 v52; // r9
  __int64 v53; // rax
  __int64 v54; // rbx
  __int64 v55; // r8
  __int64 v56; // r9
  __int64 v57; // rax
  __int64 v58; // rax
  __int64 v59; // r9
  __int64 v60; // rdx
  __int64 v61; // r14
  __int64 v62; // rax
  unsigned __int8 *v63; // rax
  __int64 v64; // rbx
  __int64 v65; // r8
  __int64 v66; // r9
  __int64 v67; // rax
  __int64 v68; // rax
  __int64 v69; // rbx
  __int64 v70; // r8
  __int64 v71; // r9
  __int64 v72; // rax
  __int64 v73; // rax
  __int64 v74; // rax
  __int64 v75; // rax
  __int64 v76; // rax
  __int64 v78; // [rsp+8h] [rbp-218h]
  __int64 v80; // [rsp+28h] [rbp-1F8h]
  __int64 v81; // [rsp+38h] [rbp-1E8h] BYREF
  unsigned __int8 *v82; // [rsp+40h] [rbp-1E0h] BYREF
  unsigned __int8 *v83; // [rsp+48h] [rbp-1D8h]
  unsigned __int64 *v84; // [rsp+90h] [rbp-190h]
  unsigned int v85; // [rsp+98h] [rbp-188h]
  char v86; // [rsp+A0h] [rbp-180h] BYREF

  v4 = *(_QWORD *)(a3 + 64);
  if ( v4 )
  {
    v5 = sub_B14240(v4);
    v7 = v6;
    v8 = v5;
    if ( v6 != v5 )
    {
      while ( *(_BYTE *)(v8 + 32) )
      {
        v8 = *(_QWORD *)(v8 + 8);
        if ( v8 == v6 )
          goto LABEL_23;
      }
      if ( v8 != v6 )
      {
LABEL_7:
        sub_B129C0(&v82, v8);
        v9 = (__int64)v82;
        if ( v83 != v82 )
        {
          v80 = v7;
          v10 = v83;
          while ( 1 )
          {
            v11 = (unsigned __int64 *)(v9 & 0xFFFFFFFFFFFFFFF8LL);
            v12 = v9 & 0xFFFFFFFFFFFFFFF8LL;
            v13 = (v9 >> 2) & 1;
            if ( ((v9 >> 2) & 1) != 0 )
              v12 = *v11;
            v14 = *(_BYTE **)(v12 + 136);
            if ( !v14 )
              goto LABEL_16;
            if ( *v14 != 60 )
              goto LABEL_16;
            v81 = *(_QWORD *)(v12 + 136);
            if ( (unsigned int)sub_2A3A1B0(a1, (__int64)v14) != 2 )
              goto LABEL_16;
            v26 = sub_2A3BF30(a1, &v81);
            v29 = *(unsigned int *)(v26 + 112);
            if ( (_DWORD)v29 )
            {
              if ( v8 == *(_QWORD *)(*(_QWORD *)(v26 + 104) + 8 * v29 - 8) )
                goto LABEL_16;
              v30 = v29 + 1;
              if ( v29 + 1 > (unsigned __int64)*(unsigned int *)(v26 + 116) )
              {
LABEL_54:
                v78 = v26;
                sub_C8D5F0(v26 + 104, (const void *)(v26 + 120), v30, 8u, v27, v28);
                v26 = v78;
                v29 = *(unsigned int *)(v78 + 112);
              }
            }
            else
            {
              v29 = 0;
              v30 = 1;
              if ( !*(_DWORD *)(v26 + 116) )
                goto LABEL_54;
            }
            *(_QWORD *)(*(_QWORD *)(v26 + 104) + 8 * v29) = v8;
            ++*(_DWORD *)(v26 + 112);
LABEL_16:
            if ( v13 || !v11 )
            {
              v9 = (unsigned __int64)(v11 + 1) | 4;
              if ( v10 == (unsigned __int8 *)v9 )
              {
LABEL_18:
                v7 = v80;
                break;
              }
            }
            else
            {
              v9 = (__int64)(v11 + 18);
              if ( v10 == (unsigned __int8 *)(v11 + 18) )
                goto LABEL_18;
            }
          }
        }
        if ( *(_BYTE *)(v8 + 64) == 2 )
        {
          v31 = sub_B13320(v8);
          if ( v31 )
          {
            if ( *v31 == 60 )
            {
              v82 = v31;
              if ( (unsigned int)sub_2A3A1B0(a1, (__int64)v31) == 2 )
              {
                v34 = sub_2A3BF30(a1, (__int64 *)&v82);
                v35 = *(unsigned int *)(v34 + 112);
                if ( (_DWORD)v35 )
                {
                  if ( v8 == *(_QWORD *)(*(_QWORD *)(v34 + 104) + 8 * v35 - 8) )
                    goto LABEL_22;
                }
                else
                {
                  v35 = 0;
                }
                if ( v35 + 1 > (unsigned __int64)*(unsigned int *)(v34 + 116) )
                {
                  sub_C8D5F0(v34 + 104, (const void *)(v34 + 120), v35 + 1, 8u, v32, v33);
                  v35 = *(unsigned int *)(v34 + 112);
                }
                *(_QWORD *)(*(_QWORD *)(v34 + 104) + 8 * v35) = v8;
                ++*(_DWORD *)(v34 + 112);
              }
            }
          }
        }
LABEL_22:
        while ( 1 )
        {
          v8 = *(_QWORD *)(v8 + 8);
          if ( v8 == v7 )
            break;
          if ( !*(_BYTE *)(v8 + 32) )
          {
            if ( v7 != v8 )
              goto LABEL_7;
            break;
          }
        }
      }
    }
  }
LABEL_23:
  if ( *(_BYTE *)a3 == 85 )
  {
    if ( (unsigned __int8)sub_A73ED0((_QWORD *)(a3 + 72), 53) || (unsigned __int8)sub_B49560(a3, 53) )
    {
      *(_BYTE *)(a1 + 176) = 1;
      v15 = *(_BYTE *)a3;
    }
    else
    {
      v15 = *(_BYTE *)a3;
    }
    if ( v15 != 60 )
    {
      if ( v15 != 85 )
        goto LABEL_89;
      v16 = *(_QWORD *)(a3 - 32);
      if ( !v16 )
        goto LABEL_89;
      if ( !*(_BYTE *)v16
        && *(_QWORD *)(v16 + 24) == *(_QWORD *)(a3 + 80)
        && (*(_BYTE *)(v16 + 33) & 0x20) != 0
        && (unsigned int)(*(_DWORD *)(v16 + 36) - 210) <= 1 )
      {
        v82 = sub_98C100(*(_QWORD *)(a3 + 32 * (1LL - (*(_DWORD *)(a3 + 4) & 0x7FFFFFF))), 0);
        if ( v82 )
        {
          if ( (unsigned int)sub_2A3A1B0(a1, (__int64)v82) == 2 )
          {
            v53 = *(_QWORD *)(a3 - 32);
            if ( !v53 || *(_BYTE *)v53 || *(_QWORD *)(v53 + 24) != *(_QWORD *)(a3 + 80) )
              BUG();
            if ( *(_DWORD *)(v53 + 36) == 211 )
            {
              v69 = sub_2A3BF30(a1, (__int64 *)&v82);
              v72 = *(unsigned int *)(v69 + 16);
              if ( v72 + 1 > (unsigned __int64)*(unsigned int *)(v69 + 20) )
              {
                sub_C8D5F0(v69 + 8, (const void *)(v69 + 24), v72 + 1, 8u, v70, v71);
                v72 = *(unsigned int *)(v69 + 16);
              }
              *(_QWORD *)(*(_QWORD *)(v69 + 8) + 8 * v72) = a3;
              ++*(_DWORD *)(v69 + 16);
            }
            else
            {
              v54 = sub_2A3BF30(a1, (__int64 *)&v82);
              v57 = *(unsigned int *)(v54 + 48);
              if ( v57 + 1 > (unsigned __int64)*(unsigned int *)(v54 + 52) )
              {
                sub_C8D5F0(v54 + 40, (const void *)(v54 + 56), v57 + 1, 8u, v55, v56);
                v57 = *(unsigned int *)(v54 + 48);
              }
              *(_QWORD *)(*(_QWORD *)(v54 + 40) + 8 * v57) = a3;
              ++*(_DWORD *)(v54 + 48);
            }
          }
        }
        else
        {
          v68 = *(unsigned int *)(a1 + 56);
          if ( v68 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 60) )
          {
            sub_C8D5F0(a1 + 48, (const void *)(a1 + 64), v68 + 1, 8u, v51, v52);
            v68 = *(unsigned int *)(a1 + 56);
          }
          *(_QWORD *)(*(_QWORD *)(a1 + 48) + 8 * v68) = a3;
          ++*(_DWORD *)(a1 + 56);
        }
        return;
      }
      if ( *(_BYTE *)v16 || *(_QWORD *)(v16 + 24) != *(_QWORD *)(a3 + 80) || (*(_BYTE *)(v16 + 33) & 0x20) == 0 )
        goto LABEL_89;
      v17 = *(_DWORD *)(v16 + 36);
      if ( v17 > 0x45 )
      {
        if ( v17 != 71 )
          goto LABEL_89;
      }
      else if ( v17 <= 0x43 )
      {
        goto LABEL_89;
      }
      sub_B58E30(&v82, a3);
      v18 = v83;
      v19 = (__int64)v82;
      if ( v83 == v82 )
      {
LABEL_109:
        v62 = *(_QWORD *)(a3 - 32);
        if ( !v62 || *(_BYTE *)v62 || *(_QWORD *)(v62 + 24) != *(_QWORD *)(a3 + 80) )
          BUG();
        if ( *(_DWORD *)(v62 + 36) != 68 )
          goto LABEL_89;
        v63 = (unsigned __int8 *)sub_B595C0(a3);
        if ( !v63 )
          goto LABEL_89;
        if ( *v63 != 60 )
          goto LABEL_89;
        v82 = v63;
        if ( (unsigned int)sub_2A3A1B0(a1, (__int64)v63) != 2 )
          goto LABEL_89;
        v64 = sub_2A3BF30(a1, (__int64 *)&v82);
        v67 = *(unsigned int *)(v64 + 80);
        if ( !(_DWORD)v67 )
        {
          v67 = 0;
          goto LABEL_118;
        }
        if ( a3 != *(_QWORD *)(*(_QWORD *)(v64 + 72) + 8 * v67 - 8) )
        {
LABEL_118:
          if ( v67 + 1 > (unsigned __int64)*(unsigned int *)(v64 + 84) )
          {
            sub_C8D5F0(v64 + 72, (const void *)(v64 + 88), v67 + 1, 8u, v65, v66);
            v67 = *(unsigned int *)(v64 + 80);
          }
          *(_QWORD *)(*(_QWORD *)(v64 + 72) + 8 * v67) = a3;
          ++*(_DWORD *)(v64 + 80);
        }
LABEL_89:
        v47 = sub_2A3A000(a3);
        if ( v47 )
        {
          v50 = *(unsigned int *)(a1 + 104);
          if ( v50 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 108) )
          {
            sub_C8D5F0(a1 + 96, (const void *)(a1 + 112), v50 + 1, 8u, v48, v49);
            v50 = *(unsigned int *)(a1 + 104);
          }
          *(_QWORD *)(*(_QWORD *)(a1 + 96) + 8 * v50) = v47;
          ++*(_DWORD *)(a1 + 104);
        }
        return;
      }
      while ( 1 )
      {
        v21 = v19;
        v22 = (_QWORD *)(v19 & 0xFFFFFFFFFFFFFFF8LL);
        v23 = v22;
        LODWORD(v21) = (v21 >> 2) & 1;
        v24 = (unsigned int)v21;
        if ( (_DWORD)v21 )
          v23 = (_QWORD *)*v22;
        v25 = (_BYTE *)v23[17];
        if ( !v25 )
          goto LABEL_45;
        if ( *v25 != 60 )
          goto LABEL_45;
        v81 = v23[17];
        if ( (unsigned int)sub_2A3A1B0(a1, (__int64)v25) != 2 )
          goto LABEL_45;
        v58 = sub_2A3BF30(a1, &v81);
        v60 = *(unsigned int *)(v58 + 80);
        v61 = v58;
        if ( !(_DWORD)v60 )
          break;
        if ( a3 != *(_QWORD *)(*(_QWORD *)(v58 + 72) + 8 * v60 - 8) )
          goto LABEL_106;
LABEL_45:
        if ( v24 )
        {
          v19 = (unsigned __int64)(v22 + 1) | 4;
          v20 = (unsigned __int8 *)v19;
        }
        else
        {
          v20 = (unsigned __int8 *)(v22 + 18);
          v19 = (__int64)(v22 + 18);
        }
        if ( v20 == v18 )
          goto LABEL_109;
      }
      v60 = 0;
LABEL_106:
      if ( v60 + 1 > (unsigned __int64)*(unsigned int *)(v58 + 84) )
      {
        sub_C8D5F0(v58 + 72, (const void *)(v58 + 88), v60 + 1, 8u, v60 + 1, v59);
        v60 = *(unsigned int *)(v61 + 80);
      }
      *(_QWORD *)(*(_QWORD *)(v61 + 72) + 8 * v60) = a3;
      ++*(_DWORD *)(v61 + 80);
      goto LABEL_45;
    }
  }
  else if ( *(_BYTE *)a3 != 60 )
  {
    goto LABEL_89;
  }
  v81 = a3;
  v36 = sub_2A3A1B0(a1, a3);
  if ( v36 == 1 )
  {
    v43 = *a2;
    v44 = sub_B2BE50(*a2);
    if ( !sub_B6EA50(v44) )
    {
      v75 = sub_B2BE50(v43);
      v76 = sub_B6F970(v75);
      if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v76 + 48LL))(v76) )
        return;
    }
    sub_B174A0((__int64)&v82, *(_QWORD *)(a1 + 192), (__int64)"safeAlloca", 10, a3);
    sub_1049740(a2, (__int64)&v82);
    v45 = v84;
    v82 = (unsigned __int8 *)&unk_49D9D40;
    v41 = &v84[10 * v85];
    if ( v84 == v41 )
      goto LABEL_74;
    do
    {
      v41 -= 10;
      v46 = v41[4];
      if ( (unsigned __int64 *)v46 != v41 + 6 )
        j_j___libc_free_0(v46);
      if ( (unsigned __int64 *)*v41 != v41 + 2 )
        j_j___libc_free_0(*v41);
    }
    while ( v45 != v41 );
  }
  else
  {
    if ( v36 != 2 )
      return;
    v37 = v81;
    *(_QWORD *)sub_2A3BF30(a1, &v81) = v37;
    v38 = *a2;
    v39 = sub_B2BE50(*a2);
    if ( !sub_B6EA50(v39) )
    {
      v73 = sub_B2BE50(v38);
      v74 = sub_B6F970(v73);
      if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v74 + 48LL))(v74) )
        return;
    }
    sub_B176B0((__int64)&v82, *(_QWORD *)(a1 + 192), (__int64)"safeAlloca", 10, a3);
    sub_1049740(a2, (__int64)&v82);
    v40 = v84;
    v82 = (unsigned __int8 *)&unk_49D9D40;
    v41 = &v84[10 * v85];
    if ( v84 == v41 )
      goto LABEL_74;
    do
    {
      v41 -= 10;
      v42 = v41[4];
      if ( (unsigned __int64 *)v42 != v41 + 6 )
        j_j___libc_free_0(v42);
      if ( (unsigned __int64 *)*v41 != v41 + 2 )
        j_j___libc_free_0(*v41);
    }
    while ( v40 != v41 );
  }
  v41 = v84;
LABEL_74:
  if ( v41 != (unsigned __int64 *)&v86 )
    _libc_free((unsigned __int64)v41);
}
