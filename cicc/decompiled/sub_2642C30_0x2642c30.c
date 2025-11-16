// Function: sub_2642C30
// Address: 0x2642c30
//
unsigned __int64 __fastcall sub_2642C30(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  _QWORD *v6; // r13
  unsigned __int64 v7; // r15
  _QWORD *v8; // rax
  _QWORD *v9; // rdx
  __int64 v10; // rsi
  __int64 v11; // rcx
  const char *v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  _QWORD *v15; // rdx
  unsigned __int64 v16; // rdi
  _QWORD *v17; // rax
  __int64 v18; // rsi
  __int64 v19; // rcx
  char *v20; // rax
  size_t v21; // rdx
  size_t v22; // rax
  __int64 v23; // rax
  unsigned __int8 v24; // dl
  _BYTE **v25; // rax
  _BYTE *v26; // rdi
  unsigned __int64 v27; // rdx
  _QWORD *v28; // rdi
  unsigned __int64 v29; // rsi
  _QWORD *v30; // rax
  __int64 v31; // rcx
  __int64 v32; // rdx
  unsigned __int64 v33; // rcx
  const char *v34; // rax
  __int64 v35; // rdx
  int v36; // eax
  const char *v37; // rax
  size_t v38; // rdx
  _BYTE *v39; // rax
  char *v40; // rax
  size_t v41; // rdx
  size_t v42; // rax
  size_t v43; // rdx
  _BYTE *v44; // rdi
  __int64 v45; // r8
  unsigned __int64 v46; // rdx
  _QWORD *v47; // rax
  _QWORD *v48; // rdi
  unsigned __int64 result; // rax
  unsigned __int64 v50; // rax
  unsigned __int64 v51; // rdi
  unsigned __int64 v52; // rdx
  unsigned __int64 v53; // rax
  size_t v54; // rdx
  unsigned __int64 v55; // [rsp+8h] [rbp-A8h]
  _BYTE *s1; // [rsp+10h] [rbp-A0h]
  const char *s1a; // [rsp+10h] [rbp-A0h]
  unsigned __int64 n; // [rsp+18h] [rbp-98h]
  size_t na; // [rsp+18h] [rbp-98h]
  unsigned __int64 v61; // [rsp+20h] [rbp-90h]
  __int64 v63; // [rsp+28h] [rbp-88h]
  unsigned __int64 v64; // [rsp+28h] [rbp-88h]
  char v65; // [rsp+3Fh] [rbp-71h] BYREF
  void *dest; // [rsp+40h] [rbp-70h] BYREF
  size_t v67; // [rsp+48h] [rbp-68h]
  __int64 v68; // [rsp+50h] [rbp-60h] BYREF
  void *src; // [rsp+60h] [rbp-50h] BYREF
  size_t v70; // [rsp+68h] [rbp-48h]
  _QWORD v71[8]; // [rsp+70h] [rbp-40h] BYREF

  v6 = (_QWORD *)(a3 + 8);
  sub_B2F930(&src, a1);
  v7 = sub_B2F650((__int64)src, v70);
  sub_2240A30((unsigned __int64 *)&src);
  v8 = *(_QWORD **)(a3 + 16);
  if ( !v8 )
    goto LABEL_82;
  v9 = v6;
  do
  {
    while ( 1 )
    {
      v10 = v8[2];
      v11 = v8[3];
      if ( v7 <= v8[4] )
        break;
      v8 = (_QWORD *)v8[3];
      if ( !v11 )
        goto LABEL_6;
    }
    v9 = v8;
    v8 = (_QWORD *)v8[2];
  }
  while ( v10 );
LABEL_6:
  if ( v6 == v9
    || v7 < v9[4]
    || (result = *(unsigned __int8 *)(a3 + 343) | (unsigned __int64)(v9 + 4) & 0xFFFFFFFFFFFFFFF8LL,
        !(*(_BYTE *)(a3 + 343) & 0xF8 | (unsigned __int64)(v9 + 4) & 0xFFFFFFFFFFFFFFF8LL)) )
  {
LABEL_82:
    v12 = sub_BD5D20(a1);
    v14 = sub_B2F650((__int64)v12, v13);
    v15 = *(_QWORD **)(a3 + 16);
    v16 = v14;
    if ( !v15 )
      goto LABEL_15;
    v17 = v6;
    do
    {
      while ( 1 )
      {
        v18 = v15[2];
        v19 = v15[3];
        if ( v16 <= v15[4] )
          break;
        v15 = (_QWORD *)v15[3];
        if ( !v19 )
          goto LABEL_13;
      }
      v17 = v15;
      v15 = (_QWORD *)v15[2];
    }
    while ( v18 );
LABEL_13:
    if ( v6 == v17
      || v16 < v17[4]
      || (result = *(unsigned __int8 *)(a3 + 343) | (unsigned __int64)(v17 + 4) & 0xFFFFFFFFFFFFFFF8LL,
          (result & 0xFFFFFFFFFFFFFFF8LL) == 0) )
    {
LABEL_15:
      v20 = (char *)sub_BD5D20(a1);
      v70 = v21;
      src = v20;
      v22 = sub_C93460((__int64 *)&src, ".llvm.", 6u);
      if ( v22 == -1 )
      {
        s1 = src;
        n = v70;
      }
      else
      {
        if ( v70 <= v22 )
          v22 = v70;
        s1 = src;
        n = v22;
      }
      v23 = sub_B91CC0(a1, "thinlto_src_file", 0x10u);
      if ( v23 )
        goto LABEL_20;
      if ( !sub_B2FC80(a1) )
      {
        v50 = *(_QWORD *)(a2 + 208);
        v63 = *(_QWORD *)(a2 + 200);
        v61 = v50;
        sub_B2F7A0(&dest, s1, n, 7, v63, v50);
LABEL_26:
        v28 = v6;
        v29 = sub_B2F650((__int64)dest, v67);
        v30 = *(_QWORD **)(a3 + 16);
        if ( v30 )
        {
          do
          {
            while ( 1 )
            {
              v31 = v30[2];
              v32 = v30[3];
              if ( v29 <= v30[4] )
                break;
              v30 = (_QWORD *)v30[3];
              if ( !v32 )
                goto LABEL_31;
            }
            v28 = v30;
            v30 = (_QWORD *)v30[2];
          }
          while ( v31 );
LABEL_31:
          v52 = *(unsigned __int8 *)(a3 + 343);
          if ( v6 != v28 && v29 >= v28[4] )
          {
            v33 = v52 | (unsigned __int64)(v28 + 4) & 0xFFFFFFFFFFFFFFF8LL;
            if ( v52 & 0xFFFFFFFFFFFFFFF8LL | (unsigned __int64)(v28 + 4) & 0xFFFFFFFFFFFFFFF8LL )
              goto LABEL_61;
            goto LABEL_34;
          }
        }
        else
        {
          v52 = *(unsigned __int8 *)(a3 + 343);
        }
        v33 = v52;
LABEL_34:
        v55 = v33;
        v34 = sub_BD5D20(a1);
        v33 = v55;
        if ( n == v35 )
        {
          if ( !n || (v36 = memcmp(s1, v34, n), v33 = v55, !v36) )
          {
            if ( (*(_BYTE *)(a1 + 32) & 0xFu) - 7 <= 1 )
            {
              na = v33;
              v37 = sub_BD5D20(a1);
              v33 = na;
              if ( v38 )
              {
                s1a = v37;
                v39 = memchr(v37, 46, v38);
                v33 = na;
                if ( v39 )
                {
                  if ( v39 - s1a != -1 )
                  {
                    v40 = (char *)sub_BD5D20(a1);
                    v65 = 46;
                    v70 = v41;
                    src = v40;
                    v42 = sub_C93460((__int64 *)&src, &v65, 1u);
                    v43 = v70;
                    if ( v42 != -1 )
                    {
                      if ( v70 <= v42 )
                        v42 = v70;
                      v43 = v42;
                    }
                    sub_B2F7A0(&src, src, v43, 7, v63, v61);
                    v44 = dest;
                    if ( src == v71 )
                    {
                      v54 = v70;
                      if ( v70 )
                      {
                        if ( v70 == 1 )
                          *(_BYTE *)dest = v71[0];
                        else
                          memcpy(dest, src, v70);
                        v54 = v70;
                        v44 = dest;
                      }
                      v67 = v54;
                      v44[v54] = 0;
                      v44 = src;
                      goto LABEL_49;
                    }
                    if ( dest == &v68 )
                    {
                      dest = src;
                      v67 = v70;
                      v68 = v71[0];
                    }
                    else
                    {
                      v45 = v68;
                      dest = src;
                      v67 = v70;
                      v68 = v71[0];
                      if ( v44 )
                      {
                        src = v44;
                        v71[0] = v45;
LABEL_49:
                        v70 = 0;
                        *v44 = 0;
                        sub_2240A30((unsigned __int64 *)&src);
                        v46 = sub_B2F650((__int64)dest, v67);
                        v47 = *(_QWORD **)(a3 + 16);
                        if ( v47 )
                        {
                          v48 = v6;
                          do
                          {
                            if ( v46 > v47[4] )
                            {
                              v47 = (_QWORD *)v47[3];
                            }
                            else
                            {
                              v48 = v47;
                              v47 = (_QWORD *)v47[2];
                            }
                          }
                          while ( v47 );
                          v53 = 0;
                          if ( v6 != v48 && v46 >= v48[4] )
                            v53 = (unsigned __int64)(v48 + 4) & 0xFFFFFFFFFFFFFFF8LL;
                        }
                        else
                        {
                          v53 = 0;
                        }
                        v33 = v53 | *(unsigned __int8 *)(a3 + 343);
                        goto LABEL_61;
                      }
                    }
                    src = v71;
                    v44 = v71;
                    goto LABEL_49;
                  }
                }
              }
            }
          }
        }
LABEL_61:
        v64 = v33;
        sub_2240A30((unsigned __int64 *)&dest);
        return v64;
      }
      v23 = sub_B91CC0(a4, "thinlto_src_file", 0x10u);
      v51 = *(_QWORD *)(a2 + 208);
      v63 = *(_QWORD *)(a2 + 200);
      v61 = v51;
      if ( v23 )
      {
LABEL_20:
        v24 = *(_BYTE *)(v23 - 16);
        if ( (v24 & 2) != 0 )
          v25 = *(_BYTE ***)(v23 - 32);
        else
          v25 = (_BYTE **)(v23 - 8LL * ((v24 >> 2) & 0xF) - 16);
        v26 = *v25;
        if ( **v25 )
          v26 = 0;
        v63 = sub_B91420((__int64)v26);
        v61 = v27;
      }
      sub_B2F7A0(&dest, s1, n, 7, v63, v61);
      goto LABEL_26;
    }
  }
  return result;
}
