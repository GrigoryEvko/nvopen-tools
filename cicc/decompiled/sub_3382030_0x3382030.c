// Function: sub_3382030
// Address: 0x3382030
//
__int64 __fastcall sub_3382030(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v6; // r14
  __int64 v7; // r12
  __int64 v8; // rdx
  char *v9; // r14
  __int64 v10; // rax
  __int64 v11; // r12
  char *v12; // r12
  char *i; // r8
  __int64 v14; // rsi
  unsigned __int8 *v15; // rsi
  char *v16; // rax
  __int64 v17; // r12
  char *v18; // r12
  __int64 v19; // rdx
  __int64 v20; // r10
  __int64 *v21; // r8
  __int64 *v22; // r12
  unsigned __int8 *v23; // rsi
  char *v24; // r14
  __int64 v25; // rsi
  __int64 v26; // r12
  __int64 v27; // r12
  __int64 v28; // r12
  unsigned __int64 v29; // [rsp+8h] [rbp-88h]
  __int64 v30; // [rsp+8h] [rbp-88h]
  __int64 v31; // [rsp+8h] [rbp-88h]
  unsigned __int64 v32; // [rsp+8h] [rbp-88h]
  __int64 v33; // [rsp+8h] [rbp-88h]
  __int64 v34; // [rsp+10h] [rbp-80h]
  unsigned __int64 v35; // [rsp+18h] [rbp-78h]
  __int64 v36; // [rsp+18h] [rbp-78h]
  char *v37; // [rsp+18h] [rbp-78h]
  __int64 *v38; // [rsp+18h] [rbp-78h]
  __int64 v39; // [rsp+18h] [rbp-78h]
  unsigned __int64 v40; // [rsp+18h] [rbp-78h]
  __int64 v41; // [rsp+18h] [rbp-78h]
  unsigned __int64 v42; // [rsp+18h] [rbp-78h]
  unsigned __int64 v43; // [rsp+18h] [rbp-78h]
  unsigned __int64 v44; // [rsp+18h] [rbp-78h]
  unsigned __int64 v45; // [rsp+18h] [rbp-78h]
  __int64 v46; // [rsp+28h] [rbp-68h]
  __int64 v47; // [rsp+28h] [rbp-68h]
  char *v48; // [rsp+28h] [rbp-68h]
  char *v49; // [rsp+28h] [rbp-68h]
  char *v50; // [rsp+28h] [rbp-68h]
  unsigned __int64 v51; // [rsp+28h] [rbp-68h]
  __int64 v52; // [rsp+28h] [rbp-68h]
  __int64 *v53; // [rsp+28h] [rbp-68h]
  __int64 v54; // [rsp+28h] [rbp-68h]
  __int64 v55; // [rsp+28h] [rbp-68h]
  __int64 v56; // [rsp+28h] [rbp-68h]
  __int64 v57; // [rsp+28h] [rbp-68h]
  __int64 v58; // [rsp+28h] [rbp-68h]
  __int64 v59; // [rsp+28h] [rbp-68h]
  __int64 v60; // [rsp+30h] [rbp-60h]
  __int64 v61; // [rsp+38h] [rbp-58h]
  char *v62; // [rsp+38h] [rbp-58h]
  char *v63; // [rsp+38h] [rbp-58h]
  __int64 v64; // [rsp+40h] [rbp-50h] BYREF
  unsigned __int64 v65; // [rsp+48h] [rbp-48h]
  char v66; // [rsp+50h] [rbp-40h]

  v60 = *(_QWORD *)(a1 + 104);
  result = v60 + 32LL * *(unsigned int *)(a1 + 112);
  v34 = result;
  if ( result == v60 )
    return result;
  do
  {
    v6 = *(_QWORD *)(v60 + 8);
    v7 = *(_QWORD *)(v60 + 16);
    if ( v7 == v6 )
      goto LABEL_56;
    do
    {
      while ( 1 )
      {
        if ( *(_QWORD *)(v6 + 8) == a2 )
        {
          v61 = *(_QWORD *)(v6 + 16);
          sub_AF47B0((__int64)&v64, *(unsigned __int64 **)(a3 + 16), *(unsigned __int64 **)(a3 + 24));
          if ( !v66 )
            break;
          sub_AF47B0((__int64)&v64, *(unsigned __int64 **)(v61 + 16), *(unsigned __int64 **)(v61 + 24));
          if ( !v66 )
            break;
          sub_AF47B0((__int64)&v64, *(unsigned __int64 **)(a3 + 16), *(unsigned __int64 **)(a3 + 24));
          v35 = v65;
          v46 = v64;
          sub_AF47B0((__int64)&v64, *(unsigned __int64 **)(v61 + 16), *(unsigned __int64 **)(v61 + 24));
          if ( v65 < v35 + v46 && v35 < v65 + v64 )
            break;
        }
        v6 += 32;
        if ( v7 == v6 )
          goto LABEL_11;
      }
      v8 = v6;
      v6 += 32;
      sub_3381C20(a1, *(_BYTE **)v60, v8);
    }
    while ( v7 != v6 );
LABEL_11:
    v9 = *(char **)(v60 + 8);
    v62 = *(char **)(v60 + 16);
    v10 = (v62 - v9) >> 5;
    v11 = (v62 - v9) >> 7;
    if ( v11 > 0 )
    {
      v12 = &v9[128 * v11];
      while ( 1 )
      {
        if ( *((_QWORD *)v9 + 1) == a2 )
        {
          v47 = *((_QWORD *)v9 + 2);
          sub_AF47B0((__int64)&v64, *(unsigned __int64 **)(a3 + 16), *(unsigned __int64 **)(a3 + 24));
          if ( !v66 )
            goto LABEL_22;
          sub_AF47B0((__int64)&v64, *(unsigned __int64 **)(v47 + 16), *(unsigned __int64 **)(v47 + 24));
          if ( !v66 )
            goto LABEL_22;
          sub_AF47B0((__int64)&v64, *(unsigned __int64 **)(a3 + 16), *(unsigned __int64 **)(a3 + 24));
          v29 = v65;
          v36 = v64;
          sub_AF47B0((__int64)&v64, *(unsigned __int64 **)(v47 + 16), *(unsigned __int64 **)(v47 + 24));
          if ( v65 < v29 + v36 && v29 < v65 + v64 )
            goto LABEL_22;
        }
        if ( *((_QWORD *)v9 + 5) == a2 )
        {
          v54 = *((_QWORD *)v9 + 6);
          sub_AF47B0((__int64)&v64, *(unsigned __int64 **)(a3 + 16), *(unsigned __int64 **)(a3 + 24));
          if ( !v66
            || (sub_AF47B0((__int64)&v64, *(unsigned __int64 **)(v54 + 16), *(unsigned __int64 **)(v54 + 24)), !v66)
            || (sub_AF47B0((__int64)&v64, *(unsigned __int64 **)(a3 + 16), *(unsigned __int64 **)(a3 + 24)),
                v40 = v65,
                v31 = v64,
                sub_AF47B0((__int64)&v64, *(unsigned __int64 **)(v54 + 16), *(unsigned __int64 **)(v54 + 24)),
                v40 < v65 + v64)
            && v65 < v31 + v40 )
          {
            v9 += 32;
            goto LABEL_22;
          }
        }
        if ( *((_QWORD *)v9 + 9) == a2 )
        {
          v55 = *((_QWORD *)v9 + 10);
          sub_AF47B0((__int64)&v64, *(unsigned __int64 **)(a3 + 16), *(unsigned __int64 **)(a3 + 24));
          if ( !v66
            || (sub_AF47B0((__int64)&v64, *(unsigned __int64 **)(v55 + 16), *(unsigned __int64 **)(v55 + 24)), !v66)
            || (sub_AF47B0((__int64)&v64, *(unsigned __int64 **)(a3 + 16), *(unsigned __int64 **)(a3 + 24)),
                v32 = v65,
                v41 = v64,
                sub_AF47B0((__int64)&v64, *(unsigned __int64 **)(v55 + 16), *(unsigned __int64 **)(v55 + 24)),
                v65 < v32 + v41)
            && v32 < v65 + v64 )
          {
            v9 += 64;
            goto LABEL_22;
          }
        }
        if ( *((_QWORD *)v9 + 13) == a2 )
        {
          v56 = *((_QWORD *)v9 + 14);
          sub_AF47B0((__int64)&v64, *(unsigned __int64 **)(a3 + 16), *(unsigned __int64 **)(a3 + 24));
          if ( !v66
            || (sub_AF47B0((__int64)&v64, *(unsigned __int64 **)(v56 + 16), *(unsigned __int64 **)(v56 + 24)), !v66)
            || (sub_AF47B0((__int64)&v64, *(unsigned __int64 **)(a3 + 16), *(unsigned __int64 **)(a3 + 24)),
                v42 = v65,
                v33 = v64,
                sub_AF47B0((__int64)&v64, *(unsigned __int64 **)(v56 + 16), *(unsigned __int64 **)(v56 + 24)),
                v42 < v65 + v64)
            && v65 < v33 + v42 )
          {
            v9 += 96;
            goto LABEL_22;
          }
        }
        v9 += 128;
        if ( v9 == v12 )
        {
          v10 = (v62 - v9) >> 5;
          break;
        }
      }
    }
    if ( v10 == 2 )
      goto LABEL_86;
    if ( v10 != 3 )
    {
      if ( v10 != 1 || *((_QWORD *)v9 + 1) != a2 )
        goto LABEL_56;
      goto LABEL_79;
    }
    if ( *((_QWORD *)v9 + 1) != a2
      || (v28 = *((_QWORD *)v9 + 2),
          sub_AF47B0((__int64)&v64, *(unsigned __int64 **)(a3 + 16), *(unsigned __int64 **)(a3 + 24)),
          v66)
      && (sub_AF47B0((__int64)&v64, *(unsigned __int64 **)(v28 + 16), *(unsigned __int64 **)(v28 + 24)), v66)
      && ((sub_AF47B0((__int64)&v64, *(unsigned __int64 **)(a3 + 16), *(unsigned __int64 **)(a3 + 24)),
           v59 = v64,
           v45 = v65,
           sub_AF47B0((__int64)&v64, *(unsigned __int64 **)(v28 + 16), *(unsigned __int64 **)(v28 + 24)),
           v45 >= v65 + v64)
       || v65 >= v45 + v59) )
    {
      v9 += 32;
LABEL_86:
      if ( *((_QWORD *)v9 + 1) != a2
        || (v27 = *((_QWORD *)v9 + 2),
            sub_AF47B0((__int64)&v64, *(unsigned __int64 **)(a3 + 16), *(unsigned __int64 **)(a3 + 24)),
            v66)
        && (sub_AF47B0((__int64)&v64, *(unsigned __int64 **)(v27 + 16), *(unsigned __int64 **)(v27 + 24)), v66)
        && ((sub_AF47B0((__int64)&v64, *(unsigned __int64 **)(a3 + 16), *(unsigned __int64 **)(a3 + 24)),
             v58 = v64,
             v44 = v65,
             sub_AF47B0((__int64)&v64, *(unsigned __int64 **)(v27 + 16), *(unsigned __int64 **)(v27 + 24)),
             v44 >= v65 + v64)
         || v65 >= v44 + v58) )
      {
        v9 += 32;
        if ( *((_QWORD *)v9 + 1) != a2 )
          goto LABEL_56;
LABEL_79:
        v26 = *((_QWORD *)v9 + 2);
        sub_AF47B0((__int64)&v64, *(unsigned __int64 **)(a3 + 16), *(unsigned __int64 **)(a3 + 24));
        if ( v66 )
        {
          sub_AF47B0((__int64)&v64, *(unsigned __int64 **)(v26 + 16), *(unsigned __int64 **)(v26 + 24));
          if ( v66 )
          {
            sub_AF47B0((__int64)&v64, *(unsigned __int64 **)(a3 + 16), *(unsigned __int64 **)(a3 + 24));
            v57 = v64;
            v43 = v65;
            sub_AF47B0((__int64)&v64, *(unsigned __int64 **)(v26 + 16), *(unsigned __int64 **)(v26 + 24));
            if ( v43 >= v65 + v64 || v65 >= v43 + v57 )
              goto LABEL_56;
          }
        }
      }
    }
LABEL_22:
    if ( v9 == v62 )
      goto LABEL_56;
    if ( v62 == v9 + 32 )
      goto LABEL_40;
    for ( i = v9 + 56; ; i = v16 )
    {
      if ( *((_QWORD *)i - 2) == a2 )
      {
        v50 = i;
        v17 = *((_QWORD *)i - 1);
        sub_AF47B0((__int64)&v64, *(unsigned __int64 **)(a3 + 16), *(unsigned __int64 **)(a3 + 24));
        i = v50;
        if ( !v66 )
          goto LABEL_31;
        sub_AF47B0((__int64)&v64, *(unsigned __int64 **)(v17 + 16), *(unsigned __int64 **)(v17 + 24));
        i = v50;
        if ( !v66 )
          goto LABEL_31;
        v37 = v50;
        sub_AF47B0((__int64)&v64, *(unsigned __int64 **)(a3 + 16), *(unsigned __int64 **)(a3 + 24));
        v30 = v64;
        v51 = v65;
        sub_AF47B0((__int64)&v64, *(unsigned __int64 **)(v17 + 16), *(unsigned __int64 **)(v17 + 24));
        i = v37;
        if ( v51 < v65 + v64 && v65 < v30 + v51 )
          break;
      }
      *(_DWORD *)v9 = *((_DWORD *)i - 6);
      *((_QWORD *)v9 + 1) = *((_QWORD *)i - 2);
      *((_QWORD *)v9 + 2) = *((_QWORD *)i - 1);
      if ( i != v9 + 24 )
      {
        v14 = *((_QWORD *)v9 + 3);
        if ( v14 )
        {
          v48 = i;
          sub_B91220((__int64)(v9 + 24), v14);
          i = v48;
        }
        v15 = *(unsigned __int8 **)i;
        *((_QWORD *)v9 + 3) = *(_QWORD *)i;
        if ( v15 )
        {
          v49 = i;
          sub_B976B0((__int64)i, v15, (__int64)(v9 + 24));
          i = v49;
          *(_QWORD *)v49 = 0;
        }
      }
      v9 += 32;
LABEL_31:
      v16 = i + 32;
      if ( v62 == i + 8 )
        goto LABEL_39;
LABEL_32:
      ;
    }
    v16 = v37 + 32;
    if ( v62 != v37 + 8 )
      goto LABEL_32;
LABEL_39:
    if ( v62 != v9 )
    {
LABEL_40:
      v18 = *(char **)(v60 + 16);
      v19 = v18 - v62;
      if ( v62 == v18 )
      {
        v24 = &v9[v19];
        v63 = v24;
        do
        {
LABEL_52:
          v25 = *((_QWORD *)v24 + 3);
          if ( v25 )
            sub_B91220((__int64)(v24 + 24), v25);
          v24 += 32;
        }
        while ( v18 != v24 );
        *(_QWORD *)(v60 + 16) = v63;
      }
      else
      {
        v20 = v19 >> 5;
        if ( v19 > 0 )
        {
          v21 = (__int64 *)(v62 + 24);
          v22 = (__int64 *)(v9 + 24);
          do
          {
            *((_DWORD *)v22 - 6) = *((_DWORD *)v21 - 6);
            *(v22 - 2) = *(v21 - 2);
            *(v22 - 1) = *(v21 - 1);
            if ( v22 != v21 )
            {
              if ( *v22 )
              {
                v38 = v21;
                v52 = v20;
                sub_B91220((__int64)v22, *v22);
                v21 = v38;
                v20 = v52;
              }
              v23 = (unsigned __int8 *)*v21;
              *v22 = *v21;
              if ( v23 )
              {
                v39 = v20;
                v53 = v21;
                sub_B976B0((__int64)v21, v23, (__int64)v22);
                v21 = v53;
                v20 = v39;
                *v53 = 0;
              }
            }
            v21 += 4;
            v22 += 4;
            --v20;
          }
          while ( v20 );
          v18 = *(char **)(v60 + 16);
          v19 = v18 - v62;
        }
        v24 = &v9[v19];
        if ( v24 != v18 )
        {
          v63 = v24;
          goto LABEL_52;
        }
      }
    }
LABEL_56:
    v60 += 32;
    result = v60;
  }
  while ( v34 != v60 );
  return result;
}
