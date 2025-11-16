// Function: sub_354F340
// Address: 0x354f340
//
void __fastcall sub_354F340(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v7; // r13
  __int64 v8; // rbx
  __int64 v9; // rbx
  unsigned int v10; // r12d
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rbx
  __int64 v15; // rax
  __int64 v16; // r12
  __int64 v17; // r11
  __int64 v18; // r8
  unsigned int v19; // r13d
  __int64 v20; // rax
  unsigned int v21; // esi
  _DWORD *v22; // rdx
  int v23; // r13d
  unsigned int v24; // ecx
  unsigned int v25; // r8d
  int *v26; // rax
  int v27; // edi
  int v28; // r14d
  unsigned int v29; // r10d
  _DWORD *v30; // rax
  int v31; // r9d
  _DWORD *v32; // rax
  __int64 v33; // rax
  __int64 v34; // rdx
  _QWORD *v35; // r8
  __int64 v36; // rax
  __int64 v37; // r9
  __int64 v38; // r14
  __int64 v39; // r13
  __int64 v40; // rbx
  __int64 v41; // r12
  __int64 v42; // rdi
  int v43; // eax
  _DWORD *v44; // r12
  __int64 v45; // rdi
  int v46; // eax
  __int64 v47; // rax
  unsigned int v48; // r12d
  __int64 v49; // r8
  __int64 v50; // rax
  __int64 v51; // rdx
  _QWORD *v52; // r8
  int v53; // eax
  unsigned int v54; // ecx
  _DWORD *v55; // rdi
  int v56; // esi
  int v57; // eax
  int v58; // r10d
  _DWORD *v59; // r9
  __int64 v60; // rdi
  __int64 v61; // rsi
  int v62; // r10d
  unsigned int v63; // r8d
  int v64; // ecx
  int *v65; // r12
  int *v66; // rax
  int *v67; // rbx
  __int64 v68; // r8
  __int64 v69; // r14
  __int64 v70; // rax
  _QWORD *v71; // rdx
  int v72; // r10d
  __int64 v73; // [rsp+10h] [rbp-F0h]
  __int64 v74; // [rsp+18h] [rbp-E8h]
  __int64 v75; // [rsp+20h] [rbp-E0h]
  __int64 v76; // [rsp+20h] [rbp-E0h]
  int v77; // [rsp+20h] [rbp-E0h]
  __int64 v78; // [rsp+30h] [rbp-D0h]
  __int64 v79; // [rsp+48h] [rbp-B8h]
  __int64 v81; // [rsp+58h] [rbp-A8h]
  int v82; // [rsp+58h] [rbp-A8h]
  __int64 v83; // [rsp+60h] [rbp-A0h] BYREF
  _DWORD *v84; // [rsp+68h] [rbp-98h]
  __int64 v85; // [rsp+70h] [rbp-90h]
  unsigned int v86; // [rsp+78h] [rbp-88h]
  void *s; // [rsp+80h] [rbp-80h] BYREF
  __int64 v88; // [rsp+88h] [rbp-78h]
  _BYTE v89[48]; // [rsp+90h] [rbp-70h] BYREF
  int v90; // [rsp+C0h] [rbp-40h]

  v7 = *(_QWORD **)a1;
  v8 = v7[1] - *v7;
  s = v89;
  v9 = v8 >> 8;
  v88 = 0x600000000LL;
  v10 = (unsigned int)(v9 + 63) >> 6;
  if ( v10 > 6 )
  {
    sub_C8D5F0((__int64)&s, v89, v10, 8u, a5, a6);
    memset(s, 0, 8LL * v10);
    LODWORD(v88) = (unsigned int)(v9 + 63) >> 6;
    v7 = *(_QWORD **)a1;
  }
  else
  {
    if ( v10 && 8LL * v10 )
      memset(v89, 0, 8LL * v10);
    LODWORD(v88) = (unsigned int)(v9 + 63) >> 6;
  }
  v90 = v9;
  v84 = 0;
  v85 = 0;
  v86 = 0;
  v11 = v7[1];
  v83 = 0;
  v12 = (v11 - *v7) >> 8;
  if ( !(_DWORD)v12 )
  {
    v60 = 0;
    v61 = 0;
    goto LABEL_65;
  }
  v81 = 0;
  v78 = (unsigned int)(v12 - 1);
  while ( 1 )
  {
    if ( 8LL * (unsigned int)v88 )
      memset(s, 0, 8LL * (unsigned int)v88);
    v13 = sub_3545E90(*(_QWORD **)(a2 + 3464), **(_QWORD **)a1 + (v81 << 8));
    v14 = *(_QWORD *)v13;
    v15 = 32LL * *(unsigned int *)(v13 + 8);
    if ( v14 != v14 + v15 )
    {
      v16 = v14 + v15;
      v17 = 32 * v81;
      while ( ((*(__int64 *)(v14 + 8) >> 1) & 3) != 2 )
      {
LABEL_14:
        v19 = *(_DWORD *)(*(_QWORD *)v14 + 200LL);
        if ( v19 == -1 )
          goto LABEL_12;
        v20 = (*(__int64 *)(v14 + 8) >> 1) & 3;
        if ( v20 == 3 )
        {
          if ( *(_DWORD *)(v14 + 16) == 3 )
            goto LABEL_12;
LABEL_11:
          v18 = 8LL * (v19 >> 6);
          if ( (*(_QWORD *)((_BYTE *)s + v18) & (1LL << v19)) != 0 )
            goto LABEL_12;
          v33 = v17 + *(_QWORD *)(a1 + 784);
          v34 = *(unsigned int *)(v33 + 8);
          if ( v34 + 1 > (unsigned __int64)*(unsigned int *)(v33 + 12) )
          {
            v73 = v17;
            v76 = v17 + *(_QWORD *)(a1 + 784);
            sub_C8D5F0(v76, (const void *)(v33 + 16), v34 + 1, 4u, v18, v34 + 1);
            v33 = v76;
            v17 = v73;
            v18 = 8LL * (v19 >> 6);
            v34 = *(unsigned int *)(v76 + 8);
          }
          v14 += 32;
          *(_DWORD *)(*(_QWORD *)v33 + 4 * v34) = v19;
          v35 = (char *)s + v18;
          ++*(_DWORD *)(v33 + 8);
          *v35 |= 1LL << v19;
          if ( v16 == v14 )
            goto LABEL_29;
        }
        else
        {
          if ( v20 != 1 )
            goto LABEL_11;
LABEL_12:
          v14 += 32;
          if ( v16 == v14 )
            goto LABEL_29;
        }
      }
      v21 = v86;
      v22 = v84;
      v23 = *(_DWORD *)(*(_QWORD *)v14 + 200LL);
      if ( v86 )
      {
        v24 = v86 - 1;
        v25 = (v86 - 1) & (37 * v81);
        v26 = &v84[2 * v25];
        v27 = *v26;
        if ( *v26 == (_DWORD)v81 )
        {
LABEL_20:
          if ( v26 == &v84[2 * v86] )
          {
            v28 = v81;
LABEL_23:
            v29 = v24 & (37 * v23);
            v30 = &v22[2 * v29];
            v31 = *v30;
            if ( v23 == *v30 )
            {
LABEL_24:
              v32 = v30 + 1;
LABEL_25:
              *v32 = v28;
              goto LABEL_14;
            }
            v77 = 1;
            v55 = 0;
            while ( v31 != 0x7FFFFFFF )
            {
              if ( !v55 && v31 == 0x80000000 )
                v55 = v30;
              v29 = v24 & (v77 + v29);
              v30 = &v22[2 * v29];
              v31 = *v30;
              if ( v23 == *v30 )
                goto LABEL_24;
              ++v77;
            }
            if ( !v55 )
              v55 = v30;
            ++v83;
            v57 = v85 + 1;
            if ( 4 * ((int)v85 + 1) < 3 * v21 )
            {
              if ( v21 - (v57 + HIDWORD(v85)) > v21 >> 3 )
                goto LABEL_74;
              v74 = v17;
              sub_2FC3A50((__int64)&v83, v21);
              if ( !v86 )
              {
LABEL_117:
                LODWORD(v85) = v85 + 1;
                BUG();
              }
              v59 = 0;
              v17 = v74;
              v62 = 1;
              v63 = (v86 - 1) & (37 * v23);
              v55 = &v84[2 * v63];
              v64 = *v55;
              v57 = v85 + 1;
              if ( v23 == *v55 )
                goto LABEL_74;
              while ( v64 != 0x7FFFFFFF )
              {
                if ( !v59 && v64 == 0x80000000 )
                  v59 = v55;
                v63 = (v86 - 1) & (v62 + v63);
                v55 = &v84[2 * v63];
                v64 = *v55;
                if ( v23 == *v55 )
                  goto LABEL_74;
                ++v62;
              }
              goto LABEL_60;
            }
LABEL_56:
            v75 = v17;
            sub_2FC3A50((__int64)&v83, 2 * v21);
            if ( !v86 )
              goto LABEL_117;
            v17 = v75;
            v54 = (v86 - 1) & (37 * v23);
            v55 = &v84[2 * v54];
            v56 = *v55;
            v57 = v85 + 1;
            if ( v23 == *v55 )
              goto LABEL_74;
            v58 = 1;
            v59 = 0;
            while ( v56 != 0x7FFFFFFF )
            {
              if ( v56 == 0x80000000 && !v59 )
                v59 = v55;
              v54 = (v86 - 1) & (v58 + v54);
              v55 = &v84[2 * v54];
              v56 = *v55;
              if ( v23 == *v55 )
                goto LABEL_74;
              ++v58;
            }
LABEL_60:
            if ( v59 )
              v55 = v59;
LABEL_74:
            LODWORD(v85) = v57;
            if ( *v55 != 0x7FFFFFFF )
              --HIDWORD(v85);
            *v55 = v23;
            v32 = v55 + 1;
            v55[1] = 0;
            goto LABEL_25;
          }
          *v26 = 0x80000000;
          v21 = v86;
          v28 = v26[1];
          v22 = v84;
          LODWORD(v85) = v85 - 1;
          ++HIDWORD(v85);
          if ( v86 )
          {
LABEL_22:
            v24 = v21 - 1;
            goto LABEL_23;
          }
LABEL_55:
          ++v83;
          goto LABEL_56;
        }
        v53 = 1;
        while ( v27 != 0x7FFFFFFF )
        {
          v72 = v53 + 1;
          v25 = v24 & (v53 + v25);
          v26 = &v84[2 * v25];
          v27 = *v26;
          if ( *v26 == (_DWORD)v81 )
            goto LABEL_20;
          v53 = v72;
        }
      }
      v28 = v81;
      if ( v86 )
        goto LABEL_22;
      goto LABEL_55;
    }
LABEL_29:
    v36 = sub_35459D0(*(_QWORD **)(a2 + 3464), **(_QWORD **)a1 + (v81 << 8));
    v38 = *(_QWORD *)v36;
    v39 = 32 * v81;
    v40 = *(_QWORD *)v36 + 32LL * *(unsigned int *)(v36 + 8);
    if ( v40 != *(_QWORD *)v36 )
    {
      while ( 1 )
      {
        v41 = *(_QWORD *)(v38 + 8);
        v42 = **(_QWORD **)v38;
        if ( (unsigned int)*(unsigned __int16 *)(v42 + 68) - 1 <= 1
          && (*(_BYTE *)(*(_QWORD *)(v42 + 32) + 64LL) & 0x10) != 0 )
        {
          goto LABEL_38;
        }
        v43 = *(_DWORD *)(v42 + 44);
        if ( (v43 & 4) != 0 || (v43 & 8) == 0 )
        {
          if ( (*(_QWORD *)(*(_QWORD *)(v42 + 16) + 24LL) & 0x100000LL) == 0 )
            goto LABEL_33;
LABEL_38:
          if ( !(unsigned __int8)sub_3544720(a2, v38) || ((*(_BYTE *)(v38 + 8) ^ 6) & 6) != 0 )
            goto LABEL_33;
          v44 = (_DWORD *)(v41 & 0xFFFFFFFFFFFFFFF8LL);
          v45 = *(_QWORD *)v44;
          if ( (unsigned int)*(unsigned __int16 *)(*(_QWORD *)v44 + 68LL) - 1 > 1
            || (*(_BYTE *)(*(_QWORD *)(v45 + 32) + 64LL) & 8) == 0 )
          {
            v46 = *(_DWORD *)(v45 + 44);
            if ( (v46 & 4) != 0 || (v46 & 8) == 0 )
              v47 = (*(_QWORD *)(*(_QWORD *)(v45 + 16) + 24LL) >> 19) & 1LL;
            else
              LOBYTE(v47) = sub_2E88A90(v45, 0x80000, 1);
            if ( !(_BYTE)v47 )
              goto LABEL_33;
          }
          v48 = v44[50];
          v37 = 1LL << v48;
          v49 = 8LL * (v48 >> 6);
          if ( (*(_QWORD *)((_BYTE *)s + v49) & (1LL << v48)) != 0 )
            goto LABEL_33;
          v50 = v39 + *(_QWORD *)(a1 + 784);
          v51 = *(unsigned int *)(v50 + 8);
          if ( v51 + 1 > (unsigned __int64)*(unsigned int *)(v50 + 12) )
          {
            v79 = v39 + *(_QWORD *)(a1 + 784);
            sub_C8D5F0(v79, (const void *)(v50 + 16), v51 + 1, 4u, v49, v37);
            v50 = v79;
            v49 = 8LL * (v48 >> 6);
            v37 = 1LL << v48;
            v51 = *(unsigned int *)(v79 + 8);
          }
          v38 += 32;
          *(_DWORD *)(*(_QWORD *)v50 + 4 * v51) = v48;
          v52 = (char *)s + v49;
          ++*(_DWORD *)(v50 + 8);
          *v52 |= v37;
          if ( v40 == v38 )
            break;
        }
        else
        {
          if ( sub_2E88A90(v42, 0x100000, 1) )
            goto LABEL_38;
LABEL_33:
          v38 += 32;
          if ( v40 == v38 )
            break;
        }
      }
    }
    if ( v78 == v81 )
      break;
    ++v81;
  }
  v60 = (__int64)v84;
  v61 = 2LL * v86;
  if ( (_DWORD)v85 )
  {
    v65 = &v84[v61];
    if ( v84 != &v84[v61] )
    {
      v66 = v84;
      while ( 1 )
      {
        v67 = v66;
        if ( (unsigned int)(*v66 + 0x7FFFFFFF) <= 0xFFFFFFFD )
          break;
        v66 += 2;
        if ( v65 == v66 )
          goto LABEL_65;
      }
      if ( v65 != v66 )
      {
        do
        {
          v68 = (unsigned int)v67[1];
          if ( (*((_QWORD *)s + ((unsigned int)v67[1] >> 6)) & (1LL << v67[1])) == 0 )
          {
            v69 = *(_QWORD *)(a1 + 784) + 32LL * *v67;
            v70 = *(unsigned int *)(v69 + 8);
            if ( v70 + 1 > (unsigned __int64)*(unsigned int *)(v69 + 12) )
            {
              v82 = v67[1];
              sub_C8D5F0(*(_QWORD *)(a1 + 784) + 32LL * *v67, (const void *)(v69 + 16), v70 + 1, 4u, v68, v37);
              v70 = *(unsigned int *)(v69 + 8);
              LODWORD(v68) = v82;
            }
            *(_DWORD *)(*(_QWORD *)v69 + 4 * v70) = v68;
            v71 = s;
            ++*(_DWORD *)(v69 + 8);
            v71[(unsigned int)v67[1] >> 6] |= 1LL << v67[1];
          }
          v67 += 2;
          if ( v67 == v65 )
            break;
          while ( (unsigned int)(*v67 + 0x7FFFFFFF) > 0xFFFFFFFD )
          {
            v67 += 2;
            if ( v65 == v67 )
              goto LABEL_96;
          }
        }
        while ( v67 != v65 );
LABEL_96:
        v60 = (__int64)v84;
        v61 = 2LL * v86;
      }
    }
  }
LABEL_65:
  sub_C7D6A0(v60, v61 * 4, 4);
  if ( s != v89 )
    _libc_free((unsigned __int64)s);
}
