// Function: sub_EA4200
// Address: 0xea4200
//
__int64 __fastcall sub_EA4200(
        __int64 a1,
        int *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        char a6,
        __int64 a7,
        unsigned __int64 a8)
{
  unsigned __int8 *v8; // r12
  __int64 v9; // rax
  unsigned __int8 v10; // dl
  __int64 v11; // r13
  __int64 v12; // rbx
  char v13; // r15
  int v14; // eax
  __int64 v15; // rcx
  char v16; // al
  unsigned __int8 *v17; // rax
  __int64 v18; // r14
  unsigned __int8 v19; // si
  unsigned __int8 v21; // al
  unsigned __int8 v22; // al
  unsigned __int64 v23; // rax
  __int64 *v24; // rax
  __int64 v25; // r15
  __int64 v26; // rbx
  __int64 v27; // r14
  unsigned __int8 *v28; // r13
  void *v29; // rdi
  unsigned __int64 v30; // r12
  void *v31; // rsi
  __int64 v32; // rax
  int v33; // ebx
  unsigned __int8 *v34; // rsi
  size_t v35; // rdx
  void *v36; // rdi
  int v37; // eax
  __int64 v38; // rbx
  int v39; // r14d
  unsigned int v40; // r15d
  size_t v41; // r12
  __int64 v42; // r15
  __int64 v43; // r13
  int i; // r14d
  __int64 v45; // r14
  size_t v46; // r9
  unsigned int v47; // r14d
  size_t v48; // rbx
  int v49; // r12d
  __int64 v50; // r15
  int v51; // r15d
  __int64 v52; // rcx
  _BYTE *v53; // rax
  __int64 v54; // r14
  void *v55; // rdi
  __int64 v56; // rax
  _BYTE *v57; // rax
  unsigned __int8 *v58; // [rsp+8h] [rbp-B8h]
  unsigned __int8 *v59; // [rsp+8h] [rbp-B8h]
  __int64 v60; // [rsp+10h] [rbp-B0h]
  __int64 v61; // [rsp+10h] [rbp-B0h]
  __int64 v62; // [rsp+10h] [rbp-B0h]
  size_t v63; // [rsp+10h] [rbp-B0h]
  size_t v64; // [rsp+10h] [rbp-B0h]
  unsigned __int8 s2; // [rsp+28h] [rbp-98h]
  unsigned __int8 *s2a; // [rsp+28h] [rbp-98h]
  void *s2b; // [rsp+28h] [rbp-98h]
  void *s2c; // [rsp+28h] [rbp-98h]
  __int64 s2d; // [rsp+28h] [rbp-98h]
  size_t s2e; // [rsp+28h] [rbp-98h]
  __int64 v74; // [rsp+38h] [rbp-88h]
  _QWORD v75[3]; // [rsp+40h] [rbp-80h] BYREF
  int v76; // [rsp+5Ch] [rbp-64h] BYREF
  int *v77[12]; // [rsp+60h] [rbp-60h] BYREF

  v77[0] = &v76;
  v8 = *(unsigned __int8 **)(a3 + 16);
  v77[1] = (int *)v75;
  v77[2] = (int *)&a7;
  v9 = *(_QWORD *)(a3 + 24);
  v75[0] = a4;
  v75[1] = a5;
  v76 = a5;
  v77[3] = (int *)a1;
  v77[4] = a2;
  v74 = v9;
  if ( v9 )
  {
    v10 = *v8;
    v11 = (__int64)a2;
    v12 = 0;
    while ( 1 )
    {
      v18 = v12 + 1;
      if ( v10 != 92 )
        break;
      if ( v74 == v18 )
      {
        v13 = 92;
        v37 = isalnum(92);
        v10 = 92;
        if ( !v37 || *(_BYTE *)(a1 + 868) )
          goto LABEL_9;
LABEL_37:
        s2c = (void *)v12;
        do
        {
          v33 = v8[v18];
          if ( !isalnum(v33) )
          {
            if ( (unsigned __int8)(v33 - 36) > 0x3Bu
              || (v32 = 0x800000000000401LL, !_bittest64(&v32, (unsigned int)(v33 - 36))) )
            {
              v12 = (__int64)s2c;
              s2d = v18 - 1;
              goto LABEL_43;
            }
          }
          ++v18;
        }
        while ( v18 );
        v12 = (__int64)s2c;
LABEL_50:
        s2d = -1;
LABEL_43:
        v34 = &v8[v12];
        v35 = v18 - v12;
        if ( !*(_BYTE *)(a1 + 871) || !v76 )
        {
LABEL_44:
          v36 = *(void **)(v11 + 32);
          if ( v35 > *(_QWORD *)(v11 + 24) - (_QWORD)v36 )
          {
            sub_CB6200(v11, v34, v35);
          }
          else if ( v35 )
          {
            s2e = v35;
            memcpy(v36, v34, v35);
            *(_QWORD *)(v11 + 32) += s2e;
          }
          v12 = v18;
          goto LABEL_11;
        }
        v60 = v18;
        v38 = v75[0];
        v39 = v76;
        v58 = v8;
        v40 = 0;
        v41 = v35;
        while ( v41 != *(_QWORD *)(v38 + 8) || v41 && memcmp(*(const void **)v38, v34, v41) )
        {
          ++v40;
          v38 += 48;
          if ( v40 == v39 )
          {
            v35 = v41;
            v18 = v60;
            v8 = v58;
            goto LABEL_44;
          }
        }
        v8 = v58;
        sub_EA3570(v77, v40);
        if ( v74 == v60 )
          goto LABEL_21;
        v10 = v58[v60];
        v12 = v60;
        if ( v10 == 38 )
        {
          v12 = s2d;
          goto LABEL_63;
        }
      }
      else
      {
        s2a = &v8[v18];
        v19 = v8[v18];
        if ( a6 && v19 == 64 )
        {
          v12 += 2;
          sub_CB59D0(v11, *(unsigned int *)(a1 + 476));
          goto LABEL_11;
        }
        if ( v19 == 43 )
        {
          v12 += 2;
          sub_CB59D0(v11, *(unsigned int *)(a3 + 84));
          goto LABEL_11;
        }
        if ( v19 != 40 || v8[v12 + 2] != 41 )
        {
          v61 = v12;
          v42 = v11;
          ++v12;
          v43 = v18;
          for ( i = v8[v18]; ; i = v8[++v12] )
          {
            if ( !isalnum((unsigned __int8)i) )
            {
              if ( (unsigned __int8)(i - 36) > 0x3Bu )
              {
                v45 = v43;
                v11 = v42;
                v46 = v12 - v45;
                goto LABEL_68;
              }
              v52 = 0x800000000000401LL;
              if ( !_bittest64(&v52, (unsigned int)(i - 36)) )
                break;
            }
            if ( v74 == v12 + 1 )
            {
              v11 = v42;
              v46 = v12 - v61;
              v12 = v74;
              goto LABEL_68;
            }
          }
          v46 = v12 - v43;
          v11 = v42;
          if ( *(_BYTE *)(a1 + 871) && v74 != v12 && (_BYTE)i == 38 )
            ++v12;
LABEL_68:
          v47 = 0;
          if ( v76 )
          {
            v62 = v12;
            v48 = v46;
            v59 = v8;
            v49 = v76;
            v50 = v75[0];
            do
            {
              if ( *(_QWORD *)(v50 + 8) == v48 )
              {
                if ( !v48 )
                {
                  v51 = v49;
                  v46 = 0;
                  v8 = v59;
                  v12 = v62;
                  if ( v47 == v51 )
                    goto LABEL_85;
                  goto LABEL_75;
                }
                if ( !memcmp(*(const void **)v50, s2a, v48) )
                {
                  v12 = v62;
                  v8 = v59;
LABEL_75:
                  sub_EA3570(v77, v47);
                  goto LABEL_11;
                }
              }
              ++v47;
              v50 += 48;
            }
            while ( v47 != v49 );
            v46 = v48;
            v8 = v59;
            v12 = v62;
          }
LABEL_85:
          v53 = *(_BYTE **)(v11 + 32);
          v54 = v11;
          if ( (unsigned __int64)v53 >= *(_QWORD *)(v11 + 24) )
          {
            v64 = v46;
            v56 = sub_CB5D20(v11, 92);
            v46 = v64;
            v54 = v56;
          }
          else
          {
            *(_QWORD *)(v11 + 32) = v53 + 1;
            *v53 = 92;
          }
          v55 = *(void **)(v54 + 32);
          if ( v46 > *(_QWORD *)(v54 + 24) - (_QWORD)v55 )
          {
            sub_CB6200(v54, s2a, v46);
          }
          else if ( v46 )
          {
            v63 = v46;
            memcpy(v55, s2a, v46);
            *(_QWORD *)(v54 + 32) += v63;
          }
LABEL_11:
          if ( v74 == v12 )
            goto LABEL_21;
          goto LABEL_12;
        }
        v12 += 3;
        if ( v74 == v12 )
          goto LABEL_21;
LABEL_12:
        v10 = v8[v12];
      }
    }
    if ( v10 == 36 )
    {
      v16 = *(_BYTE *)(a1 + 868);
      if ( v74 != v18 )
      {
        if ( v16 )
        {
          if ( v76 )
          {
            v13 = 36;
          }
          else
          {
            v21 = v8[v18];
            if ( v21 == 36 )
            {
              v57 = *(_BYTE **)(v11 + 32);
              if ( (unsigned __int64)v57 >= *(_QWORD *)(v11 + 24) )
              {
                sub_CB5D20(v11, 36);
              }
              else
              {
                *(_QWORD *)(v11 + 32) = v57 + 1;
                *v57 = 36;
              }
LABEL_63:
              v12 += 2;
              goto LABEL_11;
            }
            if ( v21 == 110 )
            {
              v12 += 2;
              sub_CB59D0(v11, a8);
              goto LABEL_11;
            }
            v22 = v21 - 48;
            v13 = 36;
            if ( v22 <= 9u )
            {
              v23 = (unsigned int)(char)v22;
              if ( v23 < a8 )
              {
                v24 = (__int64 *)(a7 + 24 * v23);
                v25 = *v24;
                if ( v24[1] != *v24 )
                {
                  s2b = (void *)v12;
                  v26 = v24[1];
                  v27 = v11;
                  v28 = v8;
                  do
                  {
                    v29 = *(void **)(v27 + 32);
                    v30 = *(_QWORD *)(v25 + 16);
                    v31 = *(void **)(v25 + 8);
                    if ( v30 <= *(_QWORD *)(v27 + 24) - (_QWORD)v29 )
                    {
                      if ( v30 )
                      {
                        memcpy(v29, v31, *(_QWORD *)(v25 + 16));
                        *(_QWORD *)(v27 + 32) += v30;
                      }
                    }
                    else
                    {
                      sub_CB6200(v27, (unsigned __int8 *)v31, *(_QWORD *)(v25 + 16));
                    }
                    v25 += 40;
                  }
                  while ( v26 != v25 );
                  v8 = v28;
                  v12 = (__int64)s2b;
                  v11 = v27;
                }
              }
              goto LABEL_63;
            }
          }
LABEL_9:
          v17 = *(unsigned __int8 **)(v11 + 32);
          ++v12;
          if ( (unsigned __int64)v17 >= *(_QWORD *)(v11 + 24) )
          {
            sub_CB5D20(v11, v13);
          }
          else
          {
            *(_QWORD *)(v11 + 32) = v17 + 1;
            *v17 = v10;
          }
          goto LABEL_11;
        }
LABEL_36:
        if ( v12 == -1 )
          goto LABEL_50;
        goto LABEL_37;
      }
      v13 = 36;
    }
    else
    {
      v13 = v10;
      s2 = v10;
      v14 = isalnum(v10);
      v10 = s2;
      if ( !v14 )
      {
        if ( (unsigned __int8)(s2 - 36) > 0x3Bu )
          goto LABEL_9;
        v15 = 0x800000000000401LL;
        if ( !_bittest64(&v15, (unsigned int)s2 - 36) )
          goto LABEL_9;
      }
      v16 = *(_BYTE *)(a1 + 868);
    }
    if ( v16 )
      goto LABEL_9;
    goto LABEL_36;
  }
LABEL_21:
  ++*(_DWORD *)(a3 + 84);
  return 0;
}
