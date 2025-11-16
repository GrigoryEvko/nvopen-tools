// Function: sub_301BCE0
// Address: 0x301bce0
//
__int64 __fastcall sub_301BCE0(__int64 a1, __int64 a2)
{
  unsigned int v2; // r12d
  __int64 v4; // rax
  int v5; // eax
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  unsigned int v9; // eax
  _QWORD *v10; // rbx
  _QWORD *v11; // r12
  __int64 v12; // rax
  unsigned __int64 *v13; // rax
  unsigned __int64 v14; // r13
  __int64 v15; // rax
  __int64 v16; // r9
  __int64 v17; // rbx
  unsigned int v18; // esi
  __int64 v19; // rdi
  __int64 v20; // r8
  int v21; // r10d
  _QWORD *v22; // rax
  unsigned int v23; // edx
  _QWORD *v24; // r12
  __int64 v25; // rcx
  unsigned __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // r12
  __int64 *v30; // r15
  _BYTE *v31; // rsi
  __int64 v32; // rax
  __int64 v33; // rsi
  __int64 v34; // rcx
  __int64 v35; // r8
  __int64 v36; // r9
  char v37; // dl
  __int64 v38; // rax
  int v39; // ecx
  __int64 v40; // rax
  __int64 v41; // r13
  __int64 v42; // r14
  __int64 v43; // rax
  __int64 v44; // rdx
  __int64 v45; // rbx
  __int64 v46; // rbx
  __int64 v47; // r12
  __int64 v48; // rbx
  __int64 v49; // rax
  unsigned int *v50; // rax
  __int64 v51; // rax
  unsigned __int8 *v52; // rax
  __int64 v53; // rdx
  unsigned __int64 v54; // rax
  __int64 v55; // rdi
  __int64 v56; // rdx
  __int64 v57; // rbx
  __int64 v58; // r12
  __int64 v59; // r8
  __int64 v60; // r9
  _QWORD *v61; // rax
  unsigned __int64 v62; // rax
  __int64 v63; // rdi
  __int64 v64; // rax
  int v65; // ecx
  int v66; // r8d
  int v67; // r8d
  unsigned int v68; // edx
  __int64 v69; // r11
  int v70; // edi
  _QWORD *v71; // rsi
  int v72; // edi
  int v73; // edi
  int v74; // esi
  unsigned int v75; // r13d
  _QWORD *v76; // rdx
  __int64 v77; // r8
  __int64 *v78; // [rsp+8h] [rbp-B8h]
  __int64 v80; // [rsp+18h] [rbp-A8h]
  __int64 *v81; // [rsp+20h] [rbp-A0h]
  __int64 v82; // [rsp+28h] [rbp-98h]
  __int64 v83; // [rsp+30h] [rbp-90h]
  __int64 v84; // [rsp+38h] [rbp-88h]
  __int64 v86; // [rsp+48h] [rbp-78h]
  __int64 v87; // [rsp+48h] [rbp-78h]
  __int64 v88; // [rsp+50h] [rbp-70h]
  __int64 v89; // [rsp+58h] [rbp-68h]
  __int64 v90; // [rsp+58h] [rbp-68h]
  __int64 v91; // [rsp+68h] [rbp-58h] BYREF
  __int64 v92; // [rsp+70h] [rbp-50h] BYREF
  __int64 v93; // [rsp+78h] [rbp-48h]
  __int64 v94; // [rsp+80h] [rbp-40h]
  int v95; // [rsp+88h] [rbp-38h]

  v2 = 0;
  if ( (*(_BYTE *)(a2 + 2) & 8) == 0 )
    return v2;
  v4 = sub_B2E500(a2);
  v5 = sub_B2A630(v4);
  *(_DWORD *)(a1 + 4) = v5;
  if ( v5 > 10 )
  {
    if ( v5 != 12 )
      return v2;
  }
  else if ( v5 <= 6 )
  {
    return v2;
  }
  *(_QWORD *)(a1 + 8) = sub_B2BEC0(a2);
  sub_F62E00(a2, 0, 0, v6, v7, v8);
  sub_B2AF20((__int64)&v92, a2);
  v9 = *(_DWORD *)(a1 + 40);
  v86 = a1 + 16;
  if ( v9 )
  {
    v10 = *(_QWORD **)(a1 + 24);
    v11 = &v10[2 * v9];
    do
    {
      if ( *v10 != -8192 && *v10 != -4096 )
      {
        v12 = v10[1];
        if ( v12 )
        {
          if ( (v12 & 4) != 0 )
          {
            v13 = (unsigned __int64 *)(v12 & 0xFFFFFFFFFFFFFFF8LL);
            v14 = (unsigned __int64)v13;
            if ( v13 )
            {
              if ( (unsigned __int64 *)*v13 != v13 + 2 )
                _libc_free(*v13);
              j_j___libc_free_0(v14);
            }
          }
        }
      }
      v10 += 2;
    }
    while ( v11 != v10 );
    v9 = *(_DWORD *)(a1 + 40);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 24), 16LL * v9, 8);
  v15 = v93;
  ++*(_QWORD *)(a1 + 16);
  ++v92;
  *(_QWORD *)(a1 + 24) = v15;
  v93 = 0;
  *(_QWORD *)(a1 + 32) = v94;
  v94 = 0;
  *(_DWORD *)(a1 + 40) = v95;
  v95 = 0;
  sub_C7D6A0(0, 0, 8);
  v80 = a2 + 72;
  v89 = *(_QWORD *)(a2 + 80);
  if ( v89 != a2 + 72 )
  {
    while ( 1 )
    {
      v17 = v89 - 24;
      if ( !v89 )
        v17 = 0;
      v18 = *(_DWORD *)(a1 + 40);
      if ( !v18 )
        break;
      v19 = *(_QWORD *)(a1 + 24);
      v20 = v18 - 1;
      v21 = 1;
      v22 = 0;
      v23 = v20 & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
      v24 = (_QWORD *)(v19 + 16LL * v23);
      v25 = *v24;
      if ( v17 != *v24 )
      {
        while ( v25 != -4096 )
        {
          if ( !v22 && v25 == -8192 )
            v22 = v24;
          v16 = (unsigned int)(v21 + 1);
          v23 = v20 & (v21 + v23);
          v24 = (_QWORD *)(v19 + 16LL * v23);
          v25 = *v24;
          if ( v17 == *v24 )
            goto LABEL_22;
          ++v21;
        }
        if ( !v22 )
          v22 = v24;
        ++*(_QWORD *)(a1 + 16);
        v65 = *(_DWORD *)(a1 + 32) + 1;
        if ( 4 * v65 < 3 * v18 )
        {
          if ( v18 - *(_DWORD *)(a1 + 36) - v65 <= v18 >> 3 )
          {
            sub_B2ACE0(v86, v18);
            v72 = *(_DWORD *)(a1 + 40);
            if ( !v72 )
            {
LABEL_139:
              ++*(_DWORD *)(a1 + 32);
              BUG();
            }
            v73 = v72 - 1;
            v16 = *(_QWORD *)(a1 + 24);
            v74 = 1;
            v75 = v73 & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
            v76 = 0;
            v65 = *(_DWORD *)(a1 + 32) + 1;
            v22 = (_QWORD *)(v16 + 16LL * v75);
            v77 = *v22;
            if ( v17 != *v22 )
            {
              while ( v77 != -4096 )
              {
                if ( v77 == -8192 && !v76 )
                  v76 = v22;
                v75 = v73 & (v74 + v75);
                v22 = (_QWORD *)(v16 + 16LL * v75);
                v77 = *v22;
                if ( v17 == *v22 )
                  goto LABEL_112;
                ++v74;
              }
              if ( v76 )
                v22 = v76;
            }
          }
          goto LABEL_112;
        }
LABEL_116:
        sub_B2ACE0(v86, 2 * v18);
        v66 = *(_DWORD *)(a1 + 40);
        if ( !v66 )
          goto LABEL_139;
        v67 = v66 - 1;
        v16 = *(_QWORD *)(a1 + 24);
        v68 = v67 & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
        v65 = *(_DWORD *)(a1 + 32) + 1;
        v22 = (_QWORD *)(v16 + 16LL * v68);
        v69 = *v22;
        if ( v17 != *v22 )
        {
          v70 = 1;
          v71 = 0;
          while ( v69 != -4096 )
          {
            if ( v69 == -8192 && !v71 )
              v71 = v22;
            v68 = v67 & (v70 + v68);
            v22 = (_QWORD *)(v16 + 16LL * v68);
            v69 = *v22;
            if ( v17 == *v22 )
              goto LABEL_112;
            ++v70;
          }
          if ( v71 )
            v22 = v71;
        }
LABEL_112:
        *(_DWORD *)(a1 + 32) = v65;
        if ( *v22 != -4096 )
          --*(_DWORD *)(a1 + 36);
        *v22 = v17;
        v22[1] = 0;
        goto LABEL_31;
      }
LABEL_22:
      v26 = v24[1] & 0xFFFFFFFFFFFFFFF8LL;
      v27 = v24[1] & 4LL;
      if ( (v24[1] & 4) != 0 )
      {
        v28 = *(_QWORD *)v26;
        v29 = *(_QWORD *)v26 + 8LL * *(unsigned int *)(v26 + 8);
        goto LABEL_24;
      }
      v28 = (__int64)(v24 + 1);
      if ( v26 )
      {
        v29 = (__int64)(v24 + 2);
LABEL_24:
        if ( v29 != v28 )
        {
          v30 = (__int64 *)v28;
          do
          {
            while ( 1 )
            {
              v91 = *v30;
              v32 = sub_301A7F0(a1 + 48, &v91, v27, v28, v20, v16);
              v92 = v17;
              v31 = *(_BYTE **)(v32 + 8);
              if ( v31 != *(_BYTE **)(v32 + 16) )
                break;
              ++v30;
              sub_F38A10(v32, v31, &v92);
              if ( (__int64 *)v29 == v30 )
                goto LABEL_31;
            }
            if ( v31 )
            {
              *(_QWORD *)v31 = v17;
              v31 = *(_BYTE **)(v32 + 8);
            }
            ++v30;
            *(_QWORD *)(v32 + 8) = v31 + 8;
          }
          while ( (__int64 *)v29 != v30 );
        }
      }
LABEL_31:
      v89 = *(_QWORD *)(v89 + 8);
      if ( v80 == v89 )
        goto LABEL_32;
    }
    ++*(_QWORD *)(a1 + 16);
    goto LABEL_116;
  }
LABEL_32:
  v33 = a2;
  sub_3017F80(a1, a2);
  if ( !byte_502A8C8 )
  {
    v37 = 1;
    if ( !*(_BYTE *)a1 )
      v37 = qword_502A708;
    v33 = a2;
    sub_301AB70(a1, a2, v37);
  }
  v2 = (unsigned __int8)byte_502A7E8;
  if ( !byte_502A7E8 )
  {
    v81 = *(__int64 **)(a1 + 80);
    v78 = &v81[4 * *(unsigned int *)(a1 + 88)];
    if ( v81 != v78 )
    {
      while ( 1 )
      {
        v38 = sub_AA4FF0(*v81);
        if ( !v38 )
          BUG();
        v39 = *(unsigned __int8 *)(v38 - 24);
        if ( (unsigned int)(v39 - 80) > 1 )
        {
          v82 = 0;
          v87 = 0;
          v83 = 0;
        }
        else
        {
          v40 = v38 - 24;
          v87 = v40;
          if ( (_BYTE)v39 == 81 )
          {
            v82 = v40;
            v83 = 0;
          }
          else
          {
            v64 = 0;
            if ( (_BYTE)v39 == 80 )
              v64 = v87;
            v82 = 0;
            v83 = v64;
          }
        }
        v34 = v81[2];
        v84 = v34;
        v90 = v81[1];
        if ( v90 != v34 )
          break;
LABEL_72:
        v81 += 4;
        if ( v78 == v81 )
          goto LABEL_73;
      }
      while ( 1 )
      {
        v41 = *(_QWORD *)v90;
        v42 = *(_QWORD *)(*(_QWORD *)v90 + 56LL);
        v88 = *(_QWORD *)v90 + 48LL;
        if ( v42 != v88 )
        {
          while ( 1 )
          {
            if ( !v42 )
              BUG();
            if ( (unsigned __int8)(*(_BYTE *)(v42 - 24) - 34) <= 0x33u )
            {
              v34 = 0x8000000000041LL;
              if ( _bittest64(&v34, (unsigned int)*(unsigned __int8 *)(v42 - 24) - 34) )
              {
                if ( *(char *)(v42 - 17) >= 0 )
                  goto LABEL_84;
                v43 = sub_BD2BC0(v42 - 24);
                v45 = v43 + v44;
                if ( *(char *)(v42 - 17) < 0 )
                  v45 -= sub_BD2BC0(v42 - 24);
                v46 = v45 >> 4;
                if ( (_DWORD)v46 )
                {
                  v47 = 0;
                  v48 = 16LL * (unsigned int)v46;
                  while ( 1 )
                  {
                    v49 = 0;
                    if ( *(char *)(v42 - 17) < 0 )
                      v49 = sub_BD2BC0(v42 - 24);
                    v50 = (unsigned int *)(v47 + v49);
                    if ( *(_DWORD *)(*(_QWORD *)v50 + 8LL) == 1 )
                      break;
                    v47 += 16;
                    if ( v48 == v47 )
                      goto LABEL_84;
                  }
                  v51 = *(_QWORD *)(v42 - 24 + 32 * (v50[2] - (unsigned __int64)(*(_DWORD *)(v42 - 20) & 0x7FFFFFF)));
                }
                else
                {
LABEL_84:
                  v51 = 0;
                }
                if ( v51 != v87 )
                {
                  v52 = sub_BD3990(*(unsigned __int8 **)(v42 - 56), v33);
                  if ( *v52 )
                    break;
                  if ( (v52[33] & 0x20) == 0
                    || (v33 = 41, !(unsigned __int8)sub_A73ED0((_QWORD *)(v42 + 48), 41))
                    && (v33 = 41, !(unsigned __int8)sub_B49560(v42 - 24, 41)) )
                  {
                    if ( **(_BYTE **)(v42 - 56) != 25 )
                      break;
                  }
                }
              }
            }
            v42 = *(_QWORD *)(v42 + 8);
            if ( v88 == v42 )
              goto LABEL_63;
          }
          if ( *(_BYTE *)(v42 - 24) == 34 )
          {
            sub_F56CD0((const char *)v41, 0, v53, v34, v35, v36);
            v61 = (_QWORD *)(*(_QWORD *)(v41 + 48) & 0xFFFFFFFFFFFFFFF8LL);
            if ( (_QWORD *)v88 == v61 )
              goto LABEL_141;
            if ( !v61 )
              BUG();
            if ( (unsigned int)*((unsigned __int8 *)v61 - 24) - 30 > 0xA )
LABEL_141:
              BUG();
            v62 = *v61 & 0xFFFFFFFFFFFFFFF8LL;
            v63 = v62 - 24;
            if ( !v62 )
              v63 = 0;
            v33 = 0;
            sub_F55BE0(v63, 0, 0, 0, v59, v60);
          }
          else
          {
            v33 = 0;
            sub_F55BE0(v42 - 24, 0, 0, 0, v35, v36);
          }
        }
LABEL_63:
        v54 = *(_QWORD *)(v41 + 48) & 0xFFFFFFFFFFFFFFF8LL;
        if ( v88 == v54 )
          goto LABEL_143;
        if ( !v54 )
          BUG();
        v55 = v54 - 24;
        if ( (unsigned int)*(unsigned __int8 *)(v54 - 24) - 30 > 0xA )
LABEL_143:
          BUG();
        v56 = *(unsigned __int8 *)(v54 - 24);
        if ( (_BYTE)v56 == 30 )
        {
          if ( v87 )
            break;
        }
        if ( (_BYTE)v56 == 38 )
        {
          v34 = v82;
          if ( *(_QWORD *)(v54 - 88) != v82 )
            break;
        }
        else
        {
          if ( (_BYTE)v56 != 37 )
          {
            if ( (_BYTE)v56 == 34 && *(_DWORD *)(a1 + 4) == 9 && v83 )
            {
              v33 = 0;
              sub_F56CD0((const char *)v41, 0, v56, v34, v35, v36);
            }
            goto LABEL_71;
          }
          if ( *(_QWORD *)(v55 - 32LL * (*(_DWORD *)(v54 - 20) & 0x7FFFFFF)) != v83 )
            break;
        }
LABEL_71:
        v90 += 8;
        if ( v84 == v90 )
          goto LABEL_72;
      }
      v33 = 0;
      sub_F55BE0(v55, 0, 0, 0, v35, v36);
      goto LABEL_71;
    }
LABEL_73:
    v57 = *(_QWORD *)(a2 + 80);
    while ( v80 != v57 )
    {
      v58 = v57;
      v57 = *(_QWORD *)(v57 + 8);
      v58 -= 24;
      sub_F61E50(v58, 0);
      sub_F5CD10(v58, 1, 0, 0);
      sub_F39690(v58, 0, 0, 0, 0, 0, 0);
    }
    v2 = 1;
    sub_F62E00(a2, 0, 0, v34, v35, v36);
  }
  return v2;
}
