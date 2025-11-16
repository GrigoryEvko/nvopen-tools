// Function: sub_2052000
// Address: 0x2052000
//
_QWORD *__fastcall sub_2052000(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *result; // rax
  __int64 v5; // rdx
  _QWORD *v6; // rcx
  _QWORD *v7; // r14
  __int64 v9; // r12
  char *v10; // rcx
  signed __int64 v11; // rax
  __int64 v12; // r14
  char *v13; // r13
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // r12
  char *v22; // rdi
  __int64 v23; // r14
  char *v24; // r15
  __int64 i; // rbx
  __int64 v26; // rsi
  unsigned __int8 *v27; // rsi
  __int64 v28; // rdx
  __int64 v29; // rax
  __int64 v30; // r13
  __int64 v31; // r12
  __int64 v32; // r12
  __int64 v33; // r12
  char *v34; // rax
  __int64 v35; // rbx
  unsigned __int64 v36; // r8
  __int64 v37; // r13
  __int64 *v38; // rbx
  unsigned __int64 v39; // r12
  unsigned __int8 *v40; // rsi
  int v41; // eax
  char *v42; // rbx
  char *v43; // r13
  __int64 v44; // rsi
  __int64 v45; // rax
  __int64 v46; // r13
  __int64 v47; // rax
  __int64 v48; // rax
  __int64 v49; // r13
  __int64 v50; // r13
  unsigned int v51; // [rsp-88h] [rbp-88h]
  unsigned int v52; // [rsp-88h] [rbp-88h]
  int v53; // [rsp-80h] [rbp-80h]
  unsigned int v54; // [rsp-80h] [rbp-80h]
  unsigned int v55; // [rsp-80h] [rbp-80h]
  int v56; // [rsp-80h] [rbp-80h]
  unsigned int v57; // [rsp-80h] [rbp-80h]
  char *v58; // [rsp-80h] [rbp-80h]
  char *v59; // [rsp-80h] [rbp-80h]
  _QWORD *v60; // [rsp-78h] [rbp-78h]
  _QWORD *v61; // [rsp-70h] [rbp-70h]
  _QWORD *v62; // [rsp-70h] [rbp-70h]
  __int64 v63; // [rsp-70h] [rbp-70h]
  unsigned int v64; // [rsp-70h] [rbp-70h]
  unsigned int v65; // [rsp-70h] [rbp-70h]
  unsigned int v66; // [rsp-70h] [rbp-70h]
  char *v67; // [rsp-68h] [rbp-68h]
  char *v68; // [rsp-60h] [rbp-60h]
  char *v69; // [rsp-60h] [rbp-60h]
  char *v70; // [rsp-60h] [rbp-60h]
  char *v71; // [rsp-60h] [rbp-60h]
  int v72; // [rsp-60h] [rbp-60h]
  char *v73; // [rsp-60h] [rbp-60h]
  char *v74; // [rsp-60h] [rbp-60h]
  int v75; // [rsp-60h] [rbp-60h]
  int v76; // [rsp-58h] [rbp-58h] BYREF
  unsigned int v77; // [rsp-50h] [rbp-50h]
  char v78; // [rsp-48h] [rbp-48h]

  result = (_QWORD *)*(unsigned int *)(a1 + 88);
  if ( (_DWORD)result )
  {
    result = *(_QWORD **)(a1 + 80);
    v5 = 4LL * *(unsigned int *)(a1 + 96);
    v6 = &result[v5];
    v60 = &result[v5];
    if ( result != &result[v5] )
    {
      while ( *result == -8 || *result == -16 )
      {
        result += 4;
        if ( v6 == result )
          return result;
      }
      if ( v60 != result )
      {
        v7 = result;
        v9 = a3;
LABEL_9:
        v10 = (char *)v7[1];
        v67 = (char *)v7[2];
        v11 = 0xAAAAAAAAAAAAAAABLL * ((v67 - v10) >> 3);
        if ( v11 >> 2 <= 0 )
          goto LABEL_73;
        v61 = v7;
        v12 = v9;
        v68 = &v10[96 * (v11 >> 2)];
        v13 = v10;
        do
        {
          v20 = *(_DWORD *)(*(_QWORD *)v13 + 20LL) & 0xFFFFFFF;
          if ( *(_QWORD *)(*(_QWORD *)(*(_QWORD *)v13 + 24 * (1 - v20)) + 24LL) == a2 )
          {
            v21 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)v13 + 24 * (2 - v20)) + 24LL);
            sub_15B1350((__int64)&v76, *(unsigned __int64 **)(v12 + 24), *(unsigned __int64 **)(v12 + 32));
            if ( !v78
              || (sub_15B1350((__int64)&v76, *(unsigned __int64 **)(v21 + 24), *(unsigned __int64 **)(v21 + 32)), !v78)
              || (sub_15B1350((__int64)&v76, *(unsigned __int64 **)(v12 + 24), *(unsigned __int64 **)(v12 + 32)),
                  v53 = v76,
                  v51 = v77,
                  sub_15B1350((__int64)&v76, *(unsigned __int64 **)(v21 + 24), *(unsigned __int64 **)(v21 + 32)),
                  v51 + v53 > v77) )
            {
              v9 = v12;
              v7 = v61;
              v10 = v13;
              goto LABEL_20;
            }
          }
          v14 = *((_QWORD *)v13 + 3);
          v15 = *(_DWORD *)(v14 + 20) & 0xFFFFFFF;
          if ( *(_QWORD *)(*(_QWORD *)(v14 + 24 * (1 - v15)) + 24LL) == a2 )
          {
            v31 = *(_QWORD *)(*(_QWORD *)(v14 + 24 * (2 - v15)) + 24LL);
            sub_15B1350((__int64)&v76, *(unsigned __int64 **)(v12 + 24), *(unsigned __int64 **)(v12 + 32));
            if ( !v78
              || (sub_15B1350((__int64)&v76, *(unsigned __int64 **)(v31 + 24), *(unsigned __int64 **)(v31 + 32)), !v78)
              || (sub_15B1350((__int64)&v76, *(unsigned __int64 **)(v12 + 24), *(unsigned __int64 **)(v12 + 32)),
                  v55 = v77,
                  sub_15B1350((__int64)&v76, *(unsigned __int64 **)(v31 + 24), *(unsigned __int64 **)(v31 + 32)),
                  v77 + v76 > v55) )
            {
              v22 = v67;
              v9 = v12;
              v7 = v61;
              v10 = v13 + 24;
              if ( v67 == v13 + 24 )
                goto LABEL_40;
              goto LABEL_21;
            }
          }
          v16 = *((_QWORD *)v13 + 6);
          v17 = *(_DWORD *)(v16 + 20) & 0xFFFFFFF;
          if ( *(_QWORD *)(*(_QWORD *)(v16 + 24 * (1 - v17)) + 24LL) == a2 )
          {
            v32 = *(_QWORD *)(*(_QWORD *)(v16 + 24 * (2 - v17)) + 24LL);
            sub_15B1350((__int64)&v76, *(unsigned __int64 **)(v12 + 24), *(unsigned __int64 **)(v12 + 32));
            if ( !v78
              || (sub_15B1350((__int64)&v76, *(unsigned __int64 **)(v32 + 24), *(unsigned __int64 **)(v32 + 32)), !v78)
              || (sub_15B1350((__int64)&v76, *(unsigned __int64 **)(v12 + 24), *(unsigned __int64 **)(v12 + 32)),
                  v56 = v76,
                  v52 = v77,
                  sub_15B1350((__int64)&v76, *(unsigned __int64 **)(v32 + 24), *(unsigned __int64 **)(v32 + 32)),
                  v52 + v56 > v77) )
            {
              v9 = v12;
              v7 = v61;
              v10 = v13 + 48;
              goto LABEL_20;
            }
          }
          v18 = *((_QWORD *)v13 + 9);
          v19 = *(_DWORD *)(v18 + 20) & 0xFFFFFFF;
          if ( *(_QWORD *)(*(_QWORD *)(v18 + 24 * (1 - v19)) + 24LL) == a2 )
          {
            v33 = *(_QWORD *)(*(_QWORD *)(v18 + 24 * (2 - v19)) + 24LL);
            sub_15B1350((__int64)&v76, *(unsigned __int64 **)(v12 + 24), *(unsigned __int64 **)(v12 + 32));
            if ( !v78
              || (sub_15B1350((__int64)&v76, *(unsigned __int64 **)(v33 + 24), *(unsigned __int64 **)(v33 + 32)), !v78)
              || (sub_15B1350((__int64)&v76, *(unsigned __int64 **)(v12 + 24), *(unsigned __int64 **)(v12 + 32)),
                  v57 = v77,
                  sub_15B1350((__int64)&v76, *(unsigned __int64 **)(v33 + 24), *(unsigned __int64 **)(v33 + 32)),
                  v77 + v76 > v57) )
            {
              v9 = v12;
              v7 = v61;
              v10 = v13 + 72;
              goto LABEL_20;
            }
          }
          v13 += 96;
        }
        while ( v68 != v13 );
        v9 = v12;
        v7 = v61;
        v10 = v13;
        v11 = 0xAAAAAAAAAAAAAAABLL * ((v67 - v13) >> 3);
LABEL_73:
        if ( v11 != 2 )
        {
          if ( v11 != 3 )
          {
            if ( v11 != 1 )
              goto LABEL_40;
LABEL_76:
            v45 = *(_DWORD *)(*(_QWORD *)v10 + 20LL) & 0xFFFFFFF;
            if ( *(_QWORD *)(*(_QWORD *)(*(_QWORD *)v10 + 24 * (1 - v45)) + 24LL) != a2 )
              goto LABEL_40;
            v71 = v10;
            v46 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)v10 + 24 * (2 - v45)) + 24LL);
            sub_15B1350((__int64)&v76, *(unsigned __int64 **)(v9 + 24), *(unsigned __int64 **)(v9 + 32));
            v10 = v71;
            if ( v78 )
            {
              sub_15B1350((__int64)&v76, *(unsigned __int64 **)(v46 + 24), *(unsigned __int64 **)(v46 + 32));
              v10 = v71;
              if ( v78 )
              {
                v58 = v71;
                sub_15B1350((__int64)&v76, *(unsigned __int64 **)(v9 + 24), *(unsigned __int64 **)(v9 + 32));
                v72 = v76;
                v64 = v77;
                sub_15B1350((__int64)&v76, *(unsigned __int64 **)(v46 + 24), *(unsigned __int64 **)(v46 + 32));
                if ( v64 + v72 <= v77 )
                  goto LABEL_40;
                v10 = v58;
              }
            }
            goto LABEL_20;
          }
          v47 = *(_DWORD *)(*(_QWORD *)v10 + 20LL) & 0xFFFFFFF;
          if ( *(_QWORD *)(*(_QWORD *)(*(_QWORD *)v10 + 24 * (1 - v47)) + 24LL) == a2 )
          {
            v74 = v10;
            v50 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)v10 + 24 * (2 - v47)) + 24LL);
            sub_15B1350((__int64)&v76, *(unsigned __int64 **)(v9 + 24), *(unsigned __int64 **)(v9 + 32));
            v10 = v74;
            if ( !v78 )
              goto LABEL_20;
            sub_15B1350((__int64)&v76, *(unsigned __int64 **)(v50 + 24), *(unsigned __int64 **)(v50 + 32));
            v10 = v74;
            if ( !v78
              || (v59 = v74,
                  sub_15B1350((__int64)&v76, *(unsigned __int64 **)(v9 + 24), *(unsigned __int64 **)(v9 + 32)),
                  v75 = v76,
                  v66 = v77,
                  sub_15B1350((__int64)&v76, *(unsigned __int64 **)(v50 + 24), *(unsigned __int64 **)(v50 + 32)),
                  v10 = v59,
                  v66 + v75 > v77) )
            {
LABEL_20:
              v22 = v67;
              if ( v67 == v10 )
                goto LABEL_40;
LABEL_21:
              if ( v22 == v10 + 24 )
              {
LABEL_56:
                v34 = (char *)v7[2];
                v35 = v34 - v67;
                if ( v67 != v34 )
                {
                  v36 = 0xAAAAAAAAAAAAAAABLL * (v35 >> 3);
                  if ( v35 > 0 )
                  {
                    v69 = v10;
                    v37 = (__int64)(v67 + 8);
                    v38 = (__int64 *)(v10 + 8);
                    v63 = v9;
                    v39 = v36;
                    do
                    {
                      *(v38 - 1) = *(_QWORD *)(v37 - 8);
                      if ( v38 != (__int64 *)v37 )
                      {
                        if ( *v38 )
                          sub_161E7C0((__int64)v38, *v38);
                        v40 = *(unsigned __int8 **)v37;
                        *v38 = *(_QWORD *)v37;
                        if ( v40 )
                        {
                          sub_1623210(v37, v40, (__int64)v38);
                          *(_QWORD *)v37 = 0;
                        }
                      }
                      v41 = *(_DWORD *)(v37 + 8);
                      v38 += 3;
                      v37 += 24;
                      *((_DWORD *)v38 - 4) = v41;
                      --v39;
                    }
                    while ( v39 );
                    v34 = (char *)v7[2];
                    v10 = v69;
                    v9 = v63;
                    v35 = v34 - v67;
                  }
                }
                v42 = &v10[v35];
                if ( v42 != v34 )
                {
                  v70 = v42;
                  v43 = v34;
                  do
                  {
                    v44 = *((_QWORD *)v42 + 1);
                    if ( v44 )
                      sub_161E7C0((__int64)(v42 + 8), v44);
                    v42 += 24;
                  }
                  while ( v43 != v42 );
                  v7[2] = v70;
                }
                goto LABEL_40;
              }
              v62 = v7;
              v23 = a2;
              v24 = v10;
              for ( i = (__int64)(v10 + 32); ; i += 24 )
              {
                v28 = *(_QWORD *)(i - 8);
                v29 = *(_DWORD *)(v28 + 20) & 0xFFFFFFF;
                if ( *(_QWORD *)(*(_QWORD *)(v28 + 24 * (1 - v29)) + 24LL) == v23 )
                {
                  v30 = *(_QWORD *)(*(_QWORD *)(v28 + 24 * (2 - v29)) + 24LL);
                  sub_15B1350((__int64)&v76, *(unsigned __int64 **)(v9 + 24), *(unsigned __int64 **)(v9 + 32));
                  if ( !v78 )
                    goto LABEL_29;
                  sub_15B1350((__int64)&v76, *(unsigned __int64 **)(v30 + 24), *(unsigned __int64 **)(v30 + 32));
                  if ( !v78 )
                    goto LABEL_29;
                  sub_15B1350((__int64)&v76, *(unsigned __int64 **)(v9 + 24), *(unsigned __int64 **)(v9 + 32));
                  v54 = v77;
                  sub_15B1350((__int64)&v76, *(unsigned __int64 **)(v30 + 24), *(unsigned __int64 **)(v30 + 32));
                  if ( v77 + v76 > v54 )
                    goto LABEL_29;
                  v28 = *(_QWORD *)(i - 8);
                }
                *(_QWORD *)v24 = v28;
                if ( v24 + 8 != (char *)i )
                {
                  v26 = *((_QWORD *)v24 + 1);
                  if ( v26 )
                    sub_161E7C0((__int64)(v24 + 8), v26);
                  v27 = *(unsigned __int8 **)i;
                  *((_QWORD *)v24 + 1) = *(_QWORD *)i;
                  if ( v27 )
                  {
                    sub_1623210(i, v27, (__int64)(v24 + 8));
                    *(_QWORD *)i = 0;
                  }
                }
                v24 += 24;
                *((_DWORD *)v24 - 2) = *(_DWORD *)(i + 8);
LABEL_29:
                if ( v67 == (char *)(i + 16) )
                {
                  v10 = v24;
                  a2 = v23;
                  v7 = v62;
                  if ( v67 != v10 )
                    goto LABEL_56;
LABEL_40:
                  result = v60;
                  v7 += 4;
                  if ( v7 == v60 )
                    return result;
                  while ( 1 )
                  {
                    result = (_QWORD *)*v7;
                    if ( *v7 != -16 && result != (_QWORD *)-8LL )
                      break;
                    v7 += 4;
                    if ( v60 == v7 )
                      return result;
                  }
                  if ( v60 == v7 )
                    return result;
                  goto LABEL_9;
                }
              }
            }
          }
          v10 += 24;
        }
        v48 = *(_DWORD *)(*(_QWORD *)v10 + 20LL) & 0xFFFFFFF;
        if ( *(_QWORD *)(*(_QWORD *)(*(_QWORD *)v10 + 24 * (1 - v48)) + 24LL) != a2 )
          goto LABEL_84;
        v73 = v10;
        v49 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)v10 + 24 * (2 - v48)) + 24LL);
        sub_15B1350((__int64)&v76, *(unsigned __int64 **)(v9 + 24), *(unsigned __int64 **)(v9 + 32));
        v10 = v73;
        if ( v78 )
        {
          sub_15B1350((__int64)&v76, *(unsigned __int64 **)(v49 + 24), *(unsigned __int64 **)(v49 + 32));
          v10 = v73;
          if ( v78 )
          {
            sub_15B1350((__int64)&v76, *(unsigned __int64 **)(v9 + 24), *(unsigned __int64 **)(v9 + 32));
            v65 = v77;
            sub_15B1350((__int64)&v76, *(unsigned __int64 **)(v49 + 24), *(unsigned __int64 **)(v49 + 32));
            v10 = v73;
            if ( v77 + v76 <= v65 )
            {
LABEL_84:
              v10 += 24;
              goto LABEL_76;
            }
          }
        }
        goto LABEL_20;
      }
    }
  }
  return result;
}
