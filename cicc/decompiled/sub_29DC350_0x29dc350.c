// Function: sub_29DC350
// Address: 0x29dc350
//
char __fastcall sub_29DC350(__int64 a1, unsigned __int8 *a2)
{
  unsigned __int64 v4; // rax
  int v5; // eax
  __int64 v6; // rcx
  unsigned __int8 v7; // al
  char v8; // dl
  int v9; // eax
  __int64 v10; // rdx
  unsigned __int8 v11; // dl
  const char *v12; // rdx
  __int64 v13; // r13
  __int64 *v14; // rax
  __int64 v15; // r13
  char v16; // al
  _QWORD *v17; // rsi
  __int64 v18; // rax
  __int64 v19; // r14
  unsigned __int64 v20; // r13
  _QWORD *v21; // rax
  _QWORD *v22; // rsi
  __int64 v23; // rcx
  __int64 v24; // rdx
  unsigned __int64 v25; // rax
  int v26; // ecx
  char v27; // al
  bool v28; // zf
  unsigned __int8 v29; // al
  char *v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rcx
  __int64 v33; // r8
  __int64 v34; // r9
  int v35; // eax
  unsigned __int8 v36; // dl
  char v37; // cl
  bool v38; // si
  _QWORD *v39; // r13
  size_t v40; // r15
  const void *v41; // rax
  __int64 v42; // rdx
  __int64 v43; // r15
  char *v44; // rax
  size_t v45; // rdx
  __int64 v46; // rax
  unsigned int v47; // esi
  __int64 v48; // rdx
  __int64 v49; // r9
  unsigned int v50; // edi
  _QWORD *v51; // rax
  _QWORD *v52; // r8
  int v53; // r15d
  int v54; // eax
  int v55; // eax
  char v56; // cl
  unsigned __int8 v57; // dl
  int v58; // edi
  int v59; // edi
  __int64 v60; // rsi
  int v61; // r8d
  unsigned int j; // eax
  __int64 v63; // r10
  int v64; // edi
  int v65; // edi
  __int64 v66; // r8
  int v67; // r9d
  unsigned int i; // eax
  _QWORD *v69; // rsi
  _QWORD *v70; // r10
  int v71; // eax
  unsigned int v72; // eax
  unsigned int v73; // eax
  __int64 v75; // [rsp+10h] [rbp-C0h]
  __int64 v76; // [rsp+10h] [rbp-C0h]
  void *s2; // [rsp+18h] [rbp-B8h]
  void *s2a; // [rsp+18h] [rbp-B8h]
  unsigned __int64 v79; // [rsp+28h] [rbp-A8h] BYREF
  _QWORD *v80; // [rsp+30h] [rbp-A0h] BYREF
  size_t n; // [rsp+38h] [rbp-98h]
  _BYTE v82[16]; // [rsp+40h] [rbp-90h] BYREF
  __int64 v83[2]; // [rsp+50h] [rbp-80h] BYREF
  __int64 v84; // [rsp+60h] [rbp-70h] BYREF
  const char *v85; // [rsp+70h] [rbp-60h] BYREF
  __int64 v86; // [rsp+78h] [rbp-58h]
  __int64 v87; // [rsp+80h] [rbp-50h] BYREF
  __int16 v88; // [rsp+90h] [rbp-40h]

  v79 = 0;
  if ( (a2[7] & 0x10) != 0 )
  {
    v19 = *(_QWORD *)(a1 + 8);
    sub_B2F930(&v85, (__int64)a2);
    v20 = sub_B2F650((__int64)v85, v86);
    if ( v85 != (const char *)&v87 )
      j_j___libc_free_0((unsigned __int64)v85);
    v21 = *(_QWORD **)(v19 + 16);
    if ( v21 )
    {
      v22 = (_QWORD *)(v19 + 8);
      do
      {
        while ( 1 )
        {
          v23 = v21[2];
          v24 = v21[3];
          if ( v20 <= v21[4] )
            break;
          v21 = (_QWORD *)v21[3];
          if ( !v24 )
            goto LABEL_42;
        }
        v22 = v21;
        v21 = (_QWORD *)v21[2];
      }
      while ( v23 );
LABEL_42:
      v25 = 0;
      if ( (_QWORD *)(v19 + 8) != v22 && v20 >= v22[4] )
        v25 = (unsigned __int64)(v22 + 4) & 0xFFFFFFFFFFFFFFF8LL;
    }
    else
    {
      v25 = 0;
    }
    v79 = *(unsigned __int8 *)(v19 + 343) | v25;
  }
  if ( !sub_B2FC80((__int64)a2) )
  {
    v4 = v79 & 0xFFFFFFFFFFFFFFF8LL;
    if ( (v79 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
    {
      if ( *(_BYTE *)(*(_QWORD *)(a1 + 8) + 337LL) )
      {
        if ( *a2 == 3 )
        {
          v12 = *(const char **)(*(_QWORD *)a1 + 168LL);
          v86 = *(_QWORD *)(*(_QWORD *)a1 + 176LL);
          v85 = v12;
          v13 = *(_QWORD *)(v4 + 32);
          v14 = sub_29DB840(*(_QWORD **)(v4 + 24), v13, (__int64)&v85);
          if ( (__int64 *)v13 != v14 )
          {
            v15 = *v14;
            if ( *v14 )
            {
              if ( *(_DWORD *)(v15 + 8) == 2 )
              {
                v16 = *(_BYTE *)(v15 + 64);
                if ( (v16 & 1) != 0 || (v16 & 2) != 0 )
                {
                  v17 = (_QWORD *)sub_BD5C60((__int64)a2);
                  *((_QWORD *)a2 + 9) = sub_A7A640((__int64 *)a2 + 9, v17, "thinlto-internalize", 0x13u, 0, 0);
                  if ( *(_BYTE *)(*(_QWORD *)(a1 + 8) + 337LL) )
                  {
                    if ( (*(_BYTE *)(v15 + 64) & 2) != 0 )
                    {
                      v18 = sub_AD6530(*((_QWORD *)a2 + 3), (__int64)v17);
                      sub_B30160((__int64)a2, v18);
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  if ( (a2[32] & 0xFu) - 7 <= 1 && sub_29DBAC0(a1, (__int64)a2, v79) )
  {
    v30 = (char *)sub_BD5D20((__int64)a2);
    if ( v30 )
    {
      v80 = v82;
      sub_29DB790((__int64 *)&v80, v30, (__int64)&v30[v31]);
    }
    else
    {
      v82[0] = 0;
      v80 = v82;
      n = 0;
    }
    sub_29DBBA0(v83, a1, (__int64)a2, v32, v33, v34);
    v85 = (const char *)v83;
    v88 = 260;
    sub_BD6B50(a2, &v85);
    if ( (__int64 *)v83[0] != &v84 )
      j_j___libc_free_0(v83[0]);
    v35 = sub_29DC260(a1, a2, 1);
    if ( (unsigned int)(v35 - 7) > 1 )
    {
      v56 = v35;
      v35 &= 0xFu;
      v37 = v56 & 0xF;
      v57 = v37 | a2[32] & 0xF0;
      a2[32] = v57;
      if ( (unsigned int)(v35 - 7) > 1 )
      {
        v38 = v37 != 9;
        if ( (v57 & 0x30) == 0 || v37 == 9 )
        {
          a2[32] = a2[32] & 0xCF | 0x10;
          goto LABEL_58;
        }
      }
    }
    else
    {
      v36 = a2[32];
      a2[33] &= 0xFCu;
      v37 = v35 & 0xF;
      a2[32] = v35 & 0xF | v36 & 0xF0;
    }
    *((_WORD *)a2 + 16) = *((_WORD *)a2 + 16) & 0xBFCF | 0x4010;
    if ( v35 == 7 )
      goto LABEL_60;
    v38 = v37 != 9;
LABEL_58:
    if ( v35 != 8 && !v38 )
    {
LABEL_61:
      v39 = (_QWORD *)sub_B326A0((__int64)a2);
      if ( v39 )
      {
        v40 = n;
        s2 = v80;
        v41 = (const void *)sub_AA8810(v39);
        if ( v40 == v42 && (!v40 || !memcmp(v41, s2, v40)) )
        {
          v43 = *(_QWORD *)a1;
          s2a = (void *)(a1 + 96);
          v44 = (char *)sub_BD5D20((__int64)a2);
          v46 = sub_BAA410(v43, v44, v45);
          v47 = *(_DWORD *)(a1 + 120);
          v48 = v46;
          if ( !v47 )
          {
            ++*(_QWORD *)(a1 + 96);
            goto LABEL_91;
          }
          v49 = *(_QWORD *)(a1 + 104);
          v6 = ((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4);
          v50 = (v47 - 1) & v6;
          v51 = (_QWORD *)(v49 + 16LL * v50);
          v52 = (_QWORD *)*v51;
          if ( v39 != (_QWORD *)*v51 )
          {
            v53 = 1;
            v6 = 0;
            while ( v52 != (_QWORD *)-4096LL )
            {
              if ( v52 == (_QWORD *)-8192LL && !v6 )
                v6 = (__int64)v51;
              v71 = v53++;
              v50 = (v47 - 1) & (v71 + v50);
              v51 = (_QWORD *)(v49 + 16LL * v50);
              v52 = (_QWORD *)*v51;
              if ( v39 == (_QWORD *)*v51 )
                goto LABEL_76;
            }
            if ( !v6 )
              v6 = (__int64)v51;
            v54 = *(_DWORD *)(a1 + 112);
            ++*(_QWORD *)(a1 + 96);
            v55 = v54 + 1;
            if ( 4 * v55 < 3 * v47 )
            {
              if ( v47 - *(_DWORD *)(a1 + 116) - v55 > v47 >> 3 )
              {
LABEL_73:
                *(_DWORD *)(a1 + 112) = v55;
                if ( *(_QWORD *)v6 != -4096 )
                  --*(_DWORD *)(a1 + 116);
                *(_QWORD *)v6 = v39;
                *(_QWORD *)(v6 + 8) = v48;
                goto LABEL_76;
              }
              v76 = v48;
              sub_26F19E0((__int64)s2a, v47);
              v64 = *(_DWORD *)(a1 + 120);
              if ( v64 )
              {
                v65 = v64 - 1;
                v6 = 0;
                v48 = v76;
                v67 = 1;
                for ( i = v65 & (((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4)); ; i = v65 & v72 )
                {
                  v66 = *(_QWORD *)(a1 + 104);
                  v69 = (_QWORD *)(v66 + 16LL * i);
                  v70 = (_QWORD *)*v69;
                  if ( v39 == (_QWORD *)*v69 )
                  {
                    v6 = v66 + 16LL * i;
                    v55 = *(_DWORD *)(a1 + 112) + 1;
                    goto LABEL_73;
                  }
                  if ( v70 == (_QWORD *)-4096LL )
                    break;
                  if ( v6 || v70 != (_QWORD *)-8192LL )
                    v69 = (_QWORD *)v6;
                  v72 = v67 + i;
                  v6 = (__int64)v69;
                  ++v67;
                }
                v55 = *(_DWORD *)(a1 + 112) + 1;
                if ( !v6 )
                  v6 = (__int64)v69;
                goto LABEL_73;
              }
LABEL_120:
              ++*(_DWORD *)(a1 + 112);
              BUG();
            }
LABEL_91:
            v75 = v48;
            sub_26F19E0((__int64)s2a, 2 * v47);
            v58 = *(_DWORD *)(a1 + 120);
            if ( v58 )
            {
              v59 = v58 - 1;
              v48 = v75;
              v60 = 0;
              v61 = 1;
              for ( j = v59 & (((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4)); ; j = v59 & v73 )
              {
                v6 = *(_QWORD *)(a1 + 104) + 16LL * j;
                v63 = *(_QWORD *)v6;
                if ( v39 == *(_QWORD **)v6 )
                {
                  v55 = *(_DWORD *)(a1 + 112) + 1;
                  goto LABEL_73;
                }
                if ( v63 == -4096 )
                  break;
                if ( v63 != -8192 || v60 )
                  v6 = v60;
                v73 = v61 + j;
                v60 = v6;
                ++v61;
              }
              v55 = *(_DWORD *)(a1 + 112) + 1;
              if ( v60 )
                v6 = v60;
              goto LABEL_73;
            }
            goto LABEL_120;
          }
        }
      }
LABEL_76:
      if ( v80 != (_QWORD *)v82 )
        j_j___libc_free_0((unsigned __int64)v80);
      goto LABEL_9;
    }
LABEL_60:
    a2[33] |= 0x40u;
    goto LABEL_61;
  }
  v5 = sub_29DC260(a1, a2, 0);
  if ( (unsigned int)(v5 - 7) <= 1 )
  {
    *((_WORD *)a2 + 16) = *((_WORD *)a2 + 16) & 0xFCC0 | v5 & 0xF;
LABEL_8:
    a2[33] |= 0x40u;
    goto LABEL_9;
  }
  v6 = v5 & 0xF;
  v11 = v6 | a2[32] & 0xF0;
  a2[32] = v11;
  if ( (v5 & 0xFu) - 7 <= 1 || (v11 & 0x30) != 0 && (_BYTE)v6 != 9 )
    goto LABEL_8;
LABEL_9:
  if ( *(_BYTE *)(a1 + 25)
    && ((v7 = a2[32], v8 = v7 & 0xF, (v7 & 0xF) == 1)
     || (sub_B2FC80((__int64)a2) || *(_QWORD *)(a1 + 16) && !(unsigned __int8)sub_29DBA40(a1, (__int64)a2))
     && (v7 = a2[32], v8 = v7 & 0xF, v6 = (v7 & 0xFu) - 7, (unsigned int)v6 > 1))
    && ((v7 & 0x30) == 0 || v8 == 9) )
  {
    a2[33] &= ~0x40u;
  }
  else if ( (v79 & 0xFFFFFFFFFFFFFFF8LL) != 0
         && (unsigned __int8)sub_BAEC50(&v79, *(_BYTE *)(*(_QWORD *)(a1 + 8) + 338LL)) )
  {
    v26 = a2[33];
    v27 = v26 & 3;
    v6 = v26 | 0x40u;
    v28 = v27 == 1;
    v29 = v6;
    if ( v28 )
      v29 = v6 & 0xFC;
    a2[33] = v29;
  }
  v9 = *a2;
  v10 = (unsigned int)(v9 - 2);
  if ( (unsigned __int8)(v9 - 2) <= 1u || !(_BYTE)v9 )
  {
    LOBYTE(v9) = a2[32] & 0xF;
    if ( (_BYTE)v9 == 1 || (LOBYTE(v9) = sub_B2FC80((__int64)a2), (_BYTE)v9) )
    {
      if ( *((_QWORD *)a2 + 6) )
        LOBYTE(v9) = (unsigned __int8)sub_B2F990((__int64)a2, 0, v10, v6);
    }
  }
  return v9;
}
