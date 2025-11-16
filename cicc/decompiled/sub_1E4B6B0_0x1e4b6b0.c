// Function: sub_1E4B6B0
// Address: 0x1e4b6b0
//
void __fastcall sub_1E4B6B0(__int64 a1, _QWORD *a2)
{
  __int64 v2; // r15
  unsigned int v3; // r8d
  __int64 v4; // r9
  _QWORD *v5; // rdx
  __int64 v6; // rax
  unsigned __int64 v7; // rax
  __int64 v8; // r14
  __int64 v9; // rsi
  __int64 *v10; // rbx
  __int64 *v11; // r12
  int v12; // r11d
  __int64 v13; // rax
  __int64 v14; // rax
  unsigned __int64 v15; // rdx
  __int64 v16; // rdi
  __int64 v17; // rdx
  _QWORD *v18; // r13
  _QWORD *v19; // rbx
  __int64 v20; // rax
  _QWORD *v21; // r15
  __int64 v22; // r13
  __int64 v23; // rdi
  __int64 v24; // rax
  __int16 v25; // dx
  unsigned __int64 v26; // rax
  __int64 v27; // rdi
  __int64 v28; // rdx
  __int16 v29; // ax
  __int64 v30; // rax
  __int64 v31; // r10
  __int64 v32; // r11
  __int64 v33; // rax
  __int64 v34; // rdx
  _QWORD *v35; // r11
  unsigned int v36; // esi
  int *v37; // rcx
  unsigned int v38; // edi
  int *v39; // rax
  int v40; // edx
  int v41; // r8d
  int v42; // edx
  int v43; // edi
  int *v44; // rax
  int v45; // r10d
  int v46; // eax
  int v47; // ecx
  int *v48; // rdi
  int *v49; // r12
  int *v50; // rax
  int *v51; // rbx
  __int64 v52; // r13
  __int64 v53; // rax
  _QWORD *v54; // rdx
  int v55; // r10d
  __int64 v56; // [rsp+8h] [rbp-C8h]
  __int64 v57; // [rsp+8h] [rbp-C8h]
  __int64 v58; // [rsp+10h] [rbp-C0h]
  __int64 v59; // [rsp+10h] [rbp-C0h]
  int v60; // [rsp+18h] [rbp-B8h]
  int v61; // [rsp+1Ch] [rbp-B4h]
  unsigned int v62; // [rsp+1Ch] [rbp-B4h]
  int v63; // [rsp+1Ch] [rbp-B4h]
  int v64; // [rsp+1Ch] [rbp-B4h]
  int v65; // [rsp+1Ch] [rbp-B4h]
  int v66; // [rsp+20h] [rbp-B0h]
  int *v67; // [rsp+20h] [rbp-B0h]
  int v68; // [rsp+20h] [rbp-B0h]
  __int64 v69; // [rsp+20h] [rbp-B0h]
  __int64 v71; // [rsp+38h] [rbp-98h]
  __int64 v72; // [rsp+40h] [rbp-90h]
  __int64 v73; // [rsp+48h] [rbp-88h]
  int v74; // [rsp+54h] [rbp-7Ch] BYREF
  int *v75; // [rsp+58h] [rbp-78h] BYREF
  void *s; // [rsp+60h] [rbp-70h] BYREF
  __int64 v77; // [rsp+68h] [rbp-68h]
  __int64 v78; // [rsp+80h] [rbp-50h] BYREF
  int *v79; // [rsp+88h] [rbp-48h]
  __int64 v80; // [rsp+90h] [rbp-40h]
  unsigned int v81; // [rsp+98h] [rbp-38h]

  v2 = a1;
  sub_1BFC1A0((__int64)&s, -252645135 * ((__int64)(*(_QWORD *)(*(_QWORD *)a1 + 8LL) - **(_QWORD **)a1) >> 4), 0);
  v5 = *(_QWORD **)a1;
  v79 = 0;
  v80 = 0;
  v81 = 0;
  v6 = v5[1];
  v78 = 0;
  v7 = 0xF0F0F0F0F0F0F0F1LL * ((v6 - *v5) >> 4);
  if ( !(_DWORD)v7 )
  {
    v48 = 0;
    goto LABEL_60;
  }
  v8 = 0;
  v73 = 0;
  v71 = (unsigned int)(v7 - 1);
  while ( 2 )
  {
    if ( v77 )
      memset(s, 0, 8 * v77);
    v9 = v8 + **(_QWORD **)v2;
    v10 = *(__int64 **)(v9 + 112);
    v11 = &v10[2 * *(unsigned int *)(v9 + 120)];
    if ( v10 != v11 )
    {
      v12 = 37 * v73;
      while ( 1 )
      {
        v14 = *v10;
        if ( ((*v10 >> 1) & 3) == 2 )
          break;
LABEL_10:
        v15 = v14 & 0xFFFFFFFFFFFFFFF8LL;
        v3 = *(_DWORD *)((v14 & 0xFFFFFFFFFFFFFFF8LL) + 192);
        if ( v3 == -1
          || ((v14 >> 1) & 3) == 1
          && **(_WORD **)(*(_QWORD *)(v15 + 8) + 16LL)
          && **(_WORD **)(*(_QWORD *)(v15 + 8) + 16LL) != 45
          || (v13 = 1LL << v3, v4 = 8LL * (v3 >> 6), (*(_QWORD *)((_BYTE *)s + v4) & (1LL << v3)) != 0) )
        {
          v10 += 2;
          if ( v11 == v10 )
            goto LABEL_18;
        }
        else
        {
          v16 = 32 * v73 + *(_QWORD *)(v2 + 824);
          v17 = *(unsigned int *)(v16 + 8);
          if ( (unsigned int)v17 >= *(_DWORD *)(v16 + 12) )
          {
            v60 = v12;
            v56 = 8LL * (v3 >> 6);
            v58 = 1LL << v3;
            v62 = v3;
            sub_16CD150(v16, (const void *)(v16 + 16), 0, 4, v3, v4);
            v12 = v60;
            v4 = v56;
            v13 = v58;
            v17 = *(unsigned int *)(v16 + 8);
            v3 = v62;
          }
          v10 += 2;
          *(_DWORD *)(*(_QWORD *)v16 + 4 * v17) = v3;
          v4 += (__int64)s;
          ++*(_DWORD *)(v16 + 8);
          *(_QWORD *)v4 |= v13;
          if ( v11 == v10 )
          {
LABEL_18:
            v9 = v8 + **(_QWORD **)v2;
            goto LABEL_19;
          }
        }
      }
      v36 = v81;
      v37 = v79;
      v74 = *(_DWORD *)((v14 & 0xFFFFFFFFFFFFFFF8LL) + 192);
      if ( v81 )
      {
        LODWORD(v4) = v81 - 1;
        v38 = v12 & (v81 - 1);
        v39 = &v79[2 * v38];
        v40 = *v39;
        if ( *v39 == (_DWORD)v73 )
        {
LABEL_48:
          if ( v39 == &v79[2 * v81] )
          {
            v41 = v73;
LABEL_51:
            v42 = v74;
            v43 = v4 & (37 * v74);
            v44 = &v37[2 * v43];
            v45 = *v44;
            if ( v74 == *v44 )
            {
LABEL_52:
              v44[1] = v41;
              v14 = *v10;
              goto LABEL_10;
            }
            v63 = 1;
            v67 = 0;
            while ( v45 != 0x7FFFFFFF )
            {
              if ( v45 == 0x80000000 )
              {
                if ( v67 )
                  v44 = v67;
                v67 = v44;
              }
              v43 = v4 & (v63 + v43);
              v44 = &v37[2 * v43];
              v45 = *v44;
              if ( v74 == *v44 )
                goto LABEL_52;
              ++v63;
            }
            if ( v67 )
              v44 = v67;
            ++v78;
            v47 = v80 + 1;
            if ( 4 * ((int)v80 + 1) < 3 * v36 )
            {
              LODWORD(v4) = v36 - (v47 + HIDWORD(v80));
              if ( (unsigned int)v4 <= v36 >> 3 )
              {
                v64 = v12;
                v68 = v41;
                sub_1E4B4F0((__int64)&v78, v36);
                sub_1E480A0((__int64)&v78, &v74, &v75);
                v44 = v75;
                v42 = v74;
                v12 = v64;
                v41 = v68;
                v47 = v80 + 1;
              }
              goto LABEL_68;
            }
LABEL_57:
            v61 = v12;
            v66 = v41;
            sub_1E4B4F0((__int64)&v78, 2 * v36);
            sub_1E480A0((__int64)&v78, &v74, &v75);
            v44 = v75;
            v42 = v74;
            v41 = v66;
            v12 = v61;
            v47 = v80 + 1;
LABEL_68:
            LODWORD(v80) = v47;
            if ( *v44 != 0x7FFFFFFF )
              --HIDWORD(v80);
            *v44 = v42;
            v44[1] = 0;
            goto LABEL_52;
          }
          *v39 = 0x80000000;
          v36 = v81;
          v41 = v39[1];
          v37 = v79;
          LODWORD(v80) = v80 - 1;
          ++HIDWORD(v80);
          if ( v81 )
          {
LABEL_50:
            LODWORD(v4) = v36 - 1;
            goto LABEL_51;
          }
LABEL_56:
          ++v78;
          goto LABEL_57;
        }
        v46 = 1;
        while ( v40 != 0x7FFFFFFF )
        {
          v55 = v46 + 1;
          v38 = v4 & (v46 + v38);
          v39 = &v79[2 * v38];
          v40 = *v39;
          if ( *v39 == (_DWORD)v73 )
            goto LABEL_48;
          v46 = v55;
        }
      }
      v41 = v73;
      if ( v81 )
        goto LABEL_50;
      goto LABEL_56;
    }
LABEL_19:
    v18 = *(_QWORD **)(v9 + 32);
    v19 = &v18[2 * *(unsigned int *)(v9 + 40)];
    if ( v18 == v19 )
      goto LABEL_44;
    v20 = v2;
    v21 = *(_QWORD **)(v9 + 32);
    v22 = v20;
    v72 = 32 * v73;
    while ( 1 )
    {
      v23 = *(_QWORD *)(v9 + 8);
      v24 = *(_QWORD *)(v23 + 16);
      if ( *(_WORD *)v24 != 1 || (*(_BYTE *)(*(_QWORD *)(v23 + 32) + 64LL) & 0x10) == 0 )
      {
        v25 = *(_WORD *)(v23 + 46);
        if ( (v25 & 4) != 0 || (v25 & 8) == 0 )
        {
          if ( (*(_QWORD *)(v24 + 8) & 0x20000LL) == 0 )
            goto LABEL_23;
        }
        else if ( !sub_1E15D00(v23, 0x20000u, 1) )
        {
          goto LABEL_23;
        }
        v9 = v8 + **(_QWORD **)v22;
      }
      if ( sub_1E42180(a2, v9, (__int64)v21, 0) && (((unsigned __int8)*v21 ^ 6) & 6) == 0 )
      {
        v26 = *v21 & 0xFFFFFFFFFFFFFFF8LL;
        v27 = *(_QWORD *)(v26 + 8);
        v28 = *(_QWORD *)(v27 + 16);
        if ( *(_WORD *)v28 == 1 && (*(_BYTE *)(*(_QWORD *)(v27 + 32) + 64LL) & 8) != 0 )
          goto LABEL_39;
        v29 = *(_WORD *)(v27 + 46);
        if ( (v29 & 4) != 0 || (v29 & 8) == 0 )
          v30 = (*(_QWORD *)(v28 + 8) >> 16) & 1LL;
        else
          LOBYTE(v30) = sub_1E15D00(v27, 0x10000u, 1);
        if ( (_BYTE)v30 )
          break;
      }
LABEL_23:
      v21 += 2;
      if ( v19 == v21 )
        goto LABEL_43;
LABEL_24:
      v9 = v8 + **(_QWORD **)v22;
    }
    v26 = *v21 & 0xFFFFFFFFFFFFFFF8LL;
LABEL_39:
    LODWORD(v4) = *(_DWORD *)(v26 + 192);
    v31 = 1LL << v4;
    v32 = 8LL * ((unsigned int)v4 >> 6);
    if ( (*(_QWORD *)((_BYTE *)s + v32) & (1LL << v4)) != 0 )
      goto LABEL_23;
    v33 = *(_QWORD *)(v22 + 824) + v72;
    v34 = *(unsigned int *)(v33 + 8);
    if ( (unsigned int)v34 >= *(_DWORD *)(v33 + 12) )
    {
      v57 = 8LL * ((unsigned int)v4 >> 6);
      v59 = 1LL << v4;
      v65 = v4;
      v69 = *(_QWORD *)(v22 + 824) + v72;
      sub_16CD150(v69, (const void *)(v33 + 16), 0, 4, v3, v4);
      v33 = v69;
      v32 = v57;
      v31 = v59;
      LODWORD(v4) = v65;
      v34 = *(unsigned int *)(v69 + 8);
    }
    v21 += 2;
    *(_DWORD *)(*(_QWORD *)v33 + 4 * v34) = v4;
    v35 = (char *)s + v32;
    ++*(_DWORD *)(v33 + 8);
    *v35 |= v31;
    if ( v19 != v21 )
      goto LABEL_24;
LABEL_43:
    v2 = v22;
LABEL_44:
    v8 += 272;
    if ( v71 != v73 )
    {
      ++v73;
      continue;
    }
    break;
  }
  v48 = v79;
  if ( (_DWORD)v80 )
  {
    v49 = &v79[2 * v81];
    if ( v49 != v79 )
    {
      v50 = v79;
      while ( 1 )
      {
        v51 = v50;
        if ( (unsigned int)(*v50 + 0x7FFFFFFF) <= 0xFFFFFFFD )
          break;
        v50 += 2;
        if ( v49 == v50 )
          goto LABEL_60;
      }
      if ( v50 != v49 )
      {
        do
        {
          if ( (*((_QWORD *)s + ((unsigned int)v51[1] >> 6)) & (1LL << v51[1])) == 0 )
          {
            v52 = *(_QWORD *)(v2 + 824) + 32LL * *v51;
            v53 = *(unsigned int *)(v52 + 8);
            if ( (unsigned int)v53 >= *(_DWORD *)(v52 + 12) )
            {
              sub_16CD150(*(_QWORD *)(v2 + 824) + 32LL * *v51, (const void *)(v52 + 16), 0, 4, v3, v4);
              v53 = *(unsigned int *)(v52 + 8);
            }
            *(_DWORD *)(*(_QWORD *)v52 + 4 * v53) = v51[1];
            v54 = s;
            ++*(_DWORD *)(v52 + 8);
            v54[(unsigned int)v51[1] >> 6] |= 1LL << v51[1];
          }
          v51 += 2;
          if ( v51 == v49 )
            break;
          while ( (unsigned int)(*v51 + 0x7FFFFFFF) > 0xFFFFFFFD )
          {
            v51 += 2;
            if ( v49 == v51 )
              goto LABEL_80;
          }
        }
        while ( v49 != v51 );
LABEL_80:
        v48 = v79;
      }
    }
  }
LABEL_60:
  j___libc_free_0(v48);
  _libc_free((unsigned __int64)s);
}
