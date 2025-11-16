// Function: sub_BE8600
// Address: 0xbe8600
//
void __fastcall sub_BE8600(__int64 *a1, _BYTE *a2, const char *a3, __int64 a4, int a5)
{
  __int64 v5; // rbx
  unsigned __int8 v6; // dl
  const char *v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r12
  __int64 v11; // rax
  __int64 v12; // r14
  __int64 v13; // rdi
  __int64 v14; // rdx
  unsigned int v15; // r14d
  const char *v16; // rax
  char v17; // r15
  const char *v18; // rax
  __int64 v19; // rdi
  unsigned int v20; // eax
  char *v21; // rbx
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  unsigned int v26; // edx
  const char *v27; // rax
  char v28; // bl
  unsigned int v29; // ebx
  const char *v30; // rax
  const char *v31; // rdi
  bool v32; // al
  const char *v33; // rax
  const char *v34; // rax
  const char *v35; // rax
  unsigned int v38; // [rsp+30h] [rbp-F0h]
  const char *v40; // [rsp+38h] [rbp-E8h] BYREF
  const char *v41; // [rsp+40h] [rbp-E0h] BYREF
  unsigned int v42; // [rsp+48h] [rbp-D8h]
  __int64 v43; // [rsp+50h] [rbp-D0h] BYREF
  unsigned int v44; // [rsp+58h] [rbp-C8h]
  const char *v45; // [rsp+60h] [rbp-C0h] BYREF
  unsigned int v46; // [rsp+68h] [rbp-B8h]
  __int64 v47; // [rsp+70h] [rbp-B0h] BYREF
  unsigned int v48; // [rsp+78h] [rbp-A8h]
  const char *v49; // [rsp+80h] [rbp-A0h] BYREF
  unsigned int v50; // [rsp+88h] [rbp-98h]
  _BYTE *v51; // [rsp+90h] [rbp-90h] BYREF
  unsigned int v52; // [rsp+98h] [rbp-88h]
  _BYTE *v53; // [rsp+A0h] [rbp-80h] BYREF
  unsigned int v54; // [rsp+A8h] [rbp-78h]
  const char *v55; // [rsp+B0h] [rbp-70h] BYREF
  unsigned int v56; // [rsp+B8h] [rbp-68h]
  const char *v57; // [rsp+C0h] [rbp-60h] BYREF
  unsigned int v58; // [rsp+C8h] [rbp-58h]
  _BYTE *v59; // [rsp+D0h] [rbp-50h]
  unsigned int v60; // [rsp+D8h] [rbp-48h]
  char v61; // [rsp+E0h] [rbp-40h]
  char v62; // [rsp+E1h] [rbp-3Fh]

  v40 = a3;
  if ( (*(a3 - 16) & 2) != 0 )
  {
    v38 = *((_DWORD *)a3 - 6);
    if ( (v38 & 1) == 0 )
      goto LABEL_3;
LABEL_157:
    v62 = 1;
    v35 = "Unfinished range!";
LABEL_161:
    v57 = v35;
    v61 = 3;
    sub_BE1BE0(a1, (__int64)&v57, &v40);
    return;
  }
  v38 = (*((_WORD *)a3 - 8) >> 6) & 0xF;
  if ( (*((_WORD *)a3 - 8) & 0x40) != 0 )
    goto LABEL_157;
LABEL_3:
  if ( v38 <= 1 )
  {
    v62 = 1;
    v35 = "It should have at least one range!";
    goto LABEL_161;
  }
  v5 = 0;
  sub_AADB10((__int64)&v49, 1u, 1);
  while ( 1 )
  {
    v6 = *(v40 - 16);
    if ( (v6 & 2) != 0 )
    {
      v7 = (const char *)*((_QWORD *)v40 - 4);
      v8 = 16 * v5;
      v9 = *(_QWORD *)&v7[16 * v5];
      if ( *(_BYTE *)v9 != 1 )
        goto LABEL_145;
    }
    else
    {
      v7 = &v40[-16 - 8LL * ((v6 >> 2) & 0xF)];
      v8 = 16 * v5;
      v9 = *(_QWORD *)&v7[16 * v5];
      if ( *(_BYTE *)v9 != 1 )
      {
LABEL_145:
        v62 = 1;
        v33 = "The lower limit must be an integer!";
        goto LABEL_146;
      }
    }
    v10 = *(_QWORD *)(v9 + 136);
    if ( *(_BYTE *)v10 != 17 )
      goto LABEL_145;
    v11 = *(_QWORD *)&v7[v8 + 8];
    if ( *(_BYTE *)v11 != 1 || (v12 = *(_QWORD *)(v11 + 136), *(_BYTE *)v12 != 17) )
    {
      v62 = 1;
      v33 = "The upper limit must be an integer!";
LABEL_146:
      v57 = v33;
      v61 = 3;
      sub_BDBF70(a1, (__int64)&v57);
      goto LABEL_58;
    }
    v13 = *(_QWORD *)(v12 + 8);
    if ( *(_QWORD *)(v10 + 8) != v13 )
    {
      v62 = 1;
      v53 = a2;
      v34 = "Range pair types must match!";
      goto LABEL_165;
    }
    if ( a5 != 2 )
      break;
    if ( !sub_BCAC40(v13, 32) )
    {
      v62 = 1;
      v53 = a2;
      v34 = "noalias.addrspace type must be i32!";
      goto LABEL_165;
    }
LABEL_15:
    v42 = *(_DWORD *)(v12 + 32);
    if ( v42 > 0x40 )
      sub_C43780(&v41, v12 + 24);
    else
      v41 = *(const char **)(v12 + 24);
    v15 = *(_DWORD *)(v10 + 32);
    v44 = v15;
    if ( v15 > 0x40 )
    {
      sub_C43780(&v43, v10 + 24);
      v15 = v44;
      if ( v44 > 0x40 )
      {
        if ( !(unsigned __int8)sub_C43C50(&v43, &v41) || v15 == (unsigned int)sub_C445E0(&v43) )
          goto LABEL_21;
        v32 = v15 == (unsigned int)sub_C444A0(&v43);
        goto LABEL_142;
      }
      v16 = (const char *)v43;
    }
    else
    {
      v16 = *(const char **)(v10 + 24);
      v43 = (__int64)v16;
    }
    if ( v16 != v41 || !v15 || v16 == (const char *)(0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v15)) )
      goto LABEL_21;
    v32 = v16 == 0;
LABEL_142:
    if ( !v32 )
    {
      v62 = 1;
      v61 = 3;
      v53 = a2;
      v57 = "The upper and lower limits cannot be the same value";
      sub_BE7760(a1, (__int64)&v57, &v53);
      goto LABEL_52;
    }
LABEL_21:
    v58 = v42;
    if ( v42 > 0x40 )
      sub_C43780(&v57, &v41);
    else
      v57 = v41;
    v48 = v44;
    if ( v44 > 0x40 )
      sub_C43780(&v47, &v43);
    else
      v47 = v43;
    sub_AADC30((__int64)&v53, (__int64)&v47, (__int64 *)&v57);
    if ( v48 > 0x40 && v47 )
      j_j___libc_free_0_0(v47);
    if ( v58 > 0x40 && v57 )
      j_j___libc_free_0_0(v57);
    if ( sub_AAF7D0((__int64)&v53) || a5 != 1 && sub_AAF760((__int64)&v53) )
    {
      v62 = 1;
      v18 = "Range must not be empty!";
      goto LABEL_46;
    }
    if ( !v5 )
      goto LABEL_68;
    sub_AB2160((__int64)&v57, (__int64)&v53, (__int64)&v49, 0);
    v17 = sub_AAF7D0((__int64)&v57);
    if ( v60 > 0x40 && v59 )
      j_j___libc_free_0_0(v59);
    if ( v58 > 0x40 && v57 )
      j_j___libc_free_0_0(v57);
    if ( !v17 )
    {
      v62 = 1;
      v18 = "Intervals are overlapping";
      goto LABEL_46;
    }
    if ( (int)sub_C4C880(&v43, &v49) <= 0 )
    {
      v62 = 1;
      v18 = "Intervals are not in order";
LABEL_46:
      v57 = v18;
      v61 = 3;
      sub_BE1BE0(a1, (__int64)&v57, &v40);
      if ( v56 > 0x40 && v55 )
        j_j___libc_free_0_0(v55);
      if ( v54 > 0x40 && v53 )
        j_j___libc_free_0_0(v53);
LABEL_52:
      if ( v44 > 0x40 && v43 )
        j_j___libc_free_0_0(v43);
      if ( v42 <= 0x40 )
        goto LABEL_58;
      v19 = (__int64)v41;
      if ( !v41 )
        goto LABEL_58;
      goto LABEL_57;
    }
    if ( v56 <= 0x40 )
    {
      if ( v55 == v49 )
        goto LABEL_45;
    }
    else if ( (unsigned __int8)sub_C43C50(&v55, &v49) )
    {
      goto LABEL_45;
    }
    if ( v54 > 0x40 )
    {
      if ( (unsigned __int8)sub_C43C50(&v53, &v51) )
        goto LABEL_45;
      goto LABEL_68;
    }
    if ( v53 == v51 )
    {
LABEL_45:
      v62 = 1;
      v18 = "Intervals are contiguous";
      goto LABEL_46;
    }
LABEL_68:
    v48 = v42;
    if ( v42 > 0x40 )
      sub_C43780(&v47, &v41);
    else
      v47 = (__int64)v41;
    v46 = v44;
    if ( v44 > 0x40 )
      sub_C43780(&v45, &v43);
    else
      v45 = (const char *)v43;
    sub_AADC30((__int64)&v57, (__int64)&v45, &v47);
    if ( v50 > 0x40 && v49 )
      j_j___libc_free_0_0(v49);
    v49 = v57;
    v20 = v58;
    v58 = 0;
    v50 = v20;
    if ( v52 > 0x40 && v51 )
    {
      j_j___libc_free_0_0(v51);
      v51 = v59;
      v52 = v60;
      if ( v58 > 0x40 && v57 )
        j_j___libc_free_0_0(v57);
    }
    else
    {
      v51 = v59;
      v52 = v60;
    }
    if ( v46 > 0x40 && v45 )
      j_j___libc_free_0_0(v45);
    if ( v48 > 0x40 && v47 )
      j_j___libc_free_0_0(v47);
    if ( v56 > 0x40 && v55 )
      j_j___libc_free_0_0(v55);
    if ( v54 > 0x40 && v53 )
      j_j___libc_free_0_0(v53);
    if ( v44 > 0x40 && v43 )
      j_j___libc_free_0_0(v43);
    if ( v42 > 0x40 && v41 )
      j_j___libc_free_0_0(v41);
    if ( ++v5 == v38 >> 1 )
    {
      if ( v38 <= 5 )
        goto LABEL_58;
      v21 = (char *)(v40 - 16);
      v22 = *(_QWORD *)sub_A17150((_BYTE *)v40 - 16);
      if ( *(_BYTE *)v22 != 1 || (v23 = *(_QWORD *)(v22 + 136), *(_BYTE *)v23 != 17) )
        BUG();
      v44 = *(_DWORD *)(v23 + 32);
      if ( v44 > 0x40 )
      {
        sub_C43780(&v43, v23 + 24);
        v21 = (char *)(v40 - 16);
      }
      else
      {
        v43 = *(_QWORD *)(v23 + 24);
      }
      v24 = *((_QWORD *)sub_A17150(v21) + 1);
      if ( *(_BYTE *)v24 != 1 || (v25 = *(_QWORD *)(v24 + 136), *(_BYTE *)v25 != 17) )
        BUG();
      v26 = *(_DWORD *)(v25 + 32);
      v46 = v26;
      if ( v26 > 0x40 )
      {
        sub_C43780(&v45, v25 + 24);
        v58 = v46;
        if ( v46 > 0x40 )
        {
          sub_C43780(&v57, &v45);
LABEL_109:
          v48 = v44;
          if ( v44 > 0x40 )
            sub_C43780(&v47, &v43);
          else
            v47 = v43;
          sub_AADC30((__int64)&v53, (__int64)&v47, (__int64 *)&v57);
          if ( v48 > 0x40 && v47 )
            j_j___libc_free_0_0(v47);
          if ( v58 > 0x40 && v57 )
            j_j___libc_free_0_0(v57);
          sub_AB2160((__int64)&v57, (__int64)&v53, (__int64)&v49, 0);
          v28 = sub_AAF7D0((__int64)&v57);
          if ( v60 > 0x40 && v59 )
            j_j___libc_free_0_0(v59);
          if ( v58 > 0x40 && v57 )
            j_j___libc_free_0_0(v57);
          if ( !v28 )
          {
            v62 = 1;
            v30 = "Intervals are overlapping";
            goto LABEL_127;
          }
          v29 = v56;
          if ( v56 <= 0x40 )
          {
            if ( v55 == v49 )
              goto LABEL_126;
          }
          else if ( (unsigned __int8)sub_C43C50(&v55, &v49) )
          {
LABEL_126:
            v62 = 1;
            v30 = "Intervals are contiguous";
LABEL_127:
            v57 = v30;
            v61 = 3;
            sub_BE1BE0(a1, (__int64)&v57, &v40);
            if ( v56 <= 0x40 || (v31 = v55) == 0 )
            {
LABEL_130:
              if ( v54 <= 0x40 )
                goto LABEL_131;
              goto LABEL_174;
            }
LABEL_129:
            j_j___libc_free_0_0(v31);
            goto LABEL_130;
          }
          if ( v54 <= 0x40 )
          {
            if ( v53 != v51 )
            {
              if ( v29 <= 0x40 )
                goto LABEL_131;
              v31 = v55;
              if ( !v55 )
                goto LABEL_131;
              goto LABEL_129;
            }
          }
          else if ( !(unsigned __int8)sub_C43C50(&v53, &v51) )
          {
            if ( v29 <= 0x40 || (v31 = v55) == 0 )
            {
LABEL_174:
              if ( v53 )
                j_j___libc_free_0_0(v53);
LABEL_131:
              if ( v46 > 0x40 && v45 )
                j_j___libc_free_0_0(v45);
              if ( v44 <= 0x40 )
                goto LABEL_58;
              v19 = v43;
              if ( !v43 )
                goto LABEL_58;
LABEL_57:
              j_j___libc_free_0_0(v19);
              goto LABEL_58;
            }
            goto LABEL_129;
          }
          goto LABEL_126;
        }
      }
      else
      {
        v27 = *(const char **)(v25 + 24);
        v58 = v26;
        v45 = v27;
      }
      v57 = v45;
      goto LABEL_109;
    }
  }
  v14 = a4;
  if ( (unsigned int)*(unsigned __int8 *)(a4 + 8) - 17 <= 1 )
    v14 = **(_QWORD **)(a4 + 16);
  if ( v13 == v14 )
    goto LABEL_15;
  v62 = 1;
  v53 = a2;
  v34 = "Range types must match instruction type!";
LABEL_165:
  v57 = v34;
  v61 = 3;
  sub_BE7760(a1, (__int64)&v57, &v53);
LABEL_58:
  if ( v52 > 0x40 && v51 )
    j_j___libc_free_0_0(v51);
  if ( v50 > 0x40 )
  {
    if ( v49 )
      j_j___libc_free_0_0(v49);
  }
}
