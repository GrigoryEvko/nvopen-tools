// Function: sub_F861D0
// Address: 0xf861d0
//
__int64 __fastcall sub_F861D0(
        __int64 *a1,
        __int64 *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 *a6,
        __int64 a7,
        __int64 a8)
{
  __int64 result; // rax
  __int64 *v9; // r15
  __int64 *v10; // r14
  __int64 *v11; // r12
  __int64 v12; // rbx
  __int64 v13; // r8
  __int64 v14; // rcx
  __int64 *v15; // rdx
  __int64 *v16; // rax
  __int64 v17; // rsi
  __int64 v18; // r13
  __int64 *v19; // r15
  __int64 *v20; // rbx
  __int64 *v21; // r14
  __int64 *v22; // rdx
  __int64 v23; // r13
  __int64 v24; // r12
  __int64 *v25; // r10
  __int64 *v26; // r11
  __int64 v27; // r12
  __int64 *v28; // rax
  int v29; // r11d
  __int64 v30; // r8
  char *v31; // r10
  __int64 *v32; // r15
  __int64 v33; // r13
  char *v34; // rax
  __int64 v35; // r8
  __int64 v36; // rdx
  __int64 v37; // rbx
  __int64 v38; // rcx
  __int64 *v39; // rdx
  __int64 *v40; // rax
  __int64 v41; // rsi
  __int64 *v42; // rax
  __int64 *v43; // r14
  __int64 *v44; // r15
  __int64 v45; // rax
  __int64 v46; // r13
  __int64 v47; // rbx
  __int64 v48; // rax
  __int64 v49; // rdx
  __int64 *v50; // rax
  __int64 *v51; // r8
  __int64 v52; // rcx
  __int64 v53; // rdx
  __int64 v54; // rcx
  int v55; // [rsp+0h] [rbp-60h]
  __int64 *v56; // [rsp+8h] [rbp-58h]
  __int64 *v57; // [rsp+8h] [rbp-58h]
  __int64 v58; // [rsp+8h] [rbp-58h]
  __int64 *v59; // [rsp+8h] [rbp-58h]
  int v60; // [rsp+8h] [rbp-58h]
  __int64 *v62; // [rsp+10h] [rbp-50h]
  __int64 v63; // [rsp+18h] [rbp-48h]
  __int64 *v64; // [rsp+18h] [rbp-48h]
  __int64 v65; // [rsp+18h] [rbp-48h]
  __int64 *v66; // [rsp+18h] [rbp-48h]
  __int64 v67; // [rsp+18h] [rbp-48h]
  __int64 *v68; // [rsp+18h] [rbp-48h]
  __int64 v69; // [rsp+20h] [rbp-40h]
  __int64 *v70; // [rsp+20h] [rbp-40h]
  __int64 v71; // [rsp+20h] [rbp-40h]
  bool v72; // [rsp+28h] [rbp-38h]
  bool v74; // [rsp+28h] [rbp-38h]

  result = a5;
  v9 = a2;
  v10 = a6;
  v11 = a1;
  if ( a7 <= a5 )
    result = a7;
  if ( result < a4 )
  {
    v12 = a5;
    if ( a7 >= a5 )
      goto LABEL_5;
    v25 = a2;
    v26 = a1;
    v27 = a4;
    while ( 1 )
    {
      if ( v27 > v12 )
      {
        v60 = (int)v26;
        v68 = v25;
        v33 = v27 / 2;
        v32 = &v26[2 * (v27 / 2)];
        v50 = sub_F7AA10(v25, a3, v32, a8);
        v31 = (char *)v68;
        v29 = v60;
        v70 = v50;
        v30 = ((char *)v50 - (char *)v68) >> 4;
      }
      else
      {
        v57 = v25;
        v64 = v26;
        v70 = &v25[2 * (v12 / 2)];
        v28 = sub_F7A8F0(v26, (__int64)v25, v70, a8);
        v29 = (int)v64;
        v30 = v12 / 2;
        v31 = (char *)v57;
        v32 = v28;
        v33 = ((char *)v28 - (char *)v64) >> 4;
      }
      v27 -= v33;
      v55 = v29;
      v65 = v30;
      v34 = sub_F85FA0((char *)v32, v31, (__int64)v70, v27, v30, a6, a7);
      v35 = v65;
      v66 = (__int64 *)v34;
      v58 = v35;
      sub_F861D0(v55, (_DWORD)v32, (_DWORD)v34, v33, v35, (_DWORD)a6, a7, a8);
      v36 = a7;
      result = (__int64)v66;
      v12 -= v58;
      if ( v12 <= a7 )
        v36 = v12;
      if ( v27 <= v36 )
        break;
      if ( v12 <= a7 )
      {
        v10 = a6;
        v9 = v70;
        v11 = v66;
LABEL_5:
        v13 = a3 - (_QWORD)v9;
        v14 = (a3 - (__int64)v9) >> 4;
        if ( a3 - (__int64)v9 <= 0 )
          return result;
        v15 = v10;
        v16 = v9;
        do
        {
          v17 = *v16;
          v15 += 2;
          v16 += 2;
          *(v15 - 2) = v17;
          *(v15 - 1) = *(v16 - 1);
          --v14;
        }
        while ( v14 );
        if ( v13 <= 0 )
          v13 = 16;
        result = (__int64)v10 + v13;
        if ( v11 == v9 )
        {
          v53 = v13 >> 4;
          v54 = a3;
          while ( 1 )
          {
            result -= 16;
            *(_QWORD *)(v54 - 16) = v17;
            v54 -= 16;
            *(_QWORD *)(v54 + 8) = *(_QWORD *)(result + 8);
            if ( !--v53 )
              break;
            v17 = *(_QWORD *)(result - 16);
          }
          return result;
        }
        if ( v10 == (__int64 *)result )
          return result;
        v18 = a3;
        v56 = v10;
        v19 = v9 - 2;
        v20 = (__int64 *)(result - 16);
        v62 = v11;
        v21 = (__int64 *)(v18 - 16);
        while ( 2 )
        {
          v23 = v19[1];
          v24 = *v20;
          v63 = *v19;
          v69 = v20[1];
          v72 = *(_BYTE *)(sub_D95540(v69) + 8) == 14;
          if ( v72 == (*(_BYTE *)(sub_D95540(v23) + 8) == 14) )
          {
            if ( v24 == v63 )
            {
              if ( sub_D969D0(v69) )
              {
                sub_D969D0(v23);
              }
              else if ( sub_D969D0(v23) )
              {
                goto LABEL_14;
              }
LABEL_20:
              *v21 = *v20;
              result = v20[1];
              v21[1] = result;
              if ( v56 == v20 )
                return result;
              v20 -= 2;
            }
            else
            {
              if ( v24 == sub_F79730(v24, v63, a8) )
                goto LABEL_20;
LABEL_14:
              v22 = v21;
              *v21 = *v19;
              v21[1] = v19[1];
              if ( v19 == v62 )
              {
                v51 = v20 + 2;
                result = ((char *)(v20 + 2) - (char *)v56) >> 4;
                if ( (char *)(v20 + 2) - (char *)v56 > 0 )
                {
                  do
                  {
                    v52 = *(v51 - 2);
                    v51 -= 2;
                    v22 -= 2;
                    *v22 = v52;
                    v22[1] = v51[1];
                    --result;
                  }
                  while ( result );
                }
                return result;
              }
              v19 -= 2;
            }
            v21 -= 2;
            continue;
          }
          break;
        }
        if ( *(_BYTE *)(sub_D95540(v69) + 8) != 14 )
          goto LABEL_20;
        goto LABEL_14;
      }
      v25 = v70;
      v26 = v66;
    }
    v10 = a6;
    v9 = v70;
    v11 = v66;
  }
  v37 = (char *)v9 - (char *)v11;
  v38 = ((char *)v9 - (char *)v11) >> 4;
  if ( (char *)v9 - (char *)v11 <= 0 )
    return result;
  v39 = v10;
  v40 = v11;
  do
  {
    v41 = *v40;
    v39 += 2;
    v40 += 2;
    *(v39 - 2) = v41;
    *(v39 - 1) = *(v40 - 1);
    --v38;
  }
  while ( v38 );
  if ( v37 <= 0 )
    v37 = 16;
  v59 = (__int64 *)((char *)v10 + v37);
  if ( (__int64 *)a3 == v9 || v10 == (__int64 *)((char *)v10 + v37) )
    goto LABEL_45;
  v42 = v10;
  v43 = v9;
  v44 = v42;
  do
  {
    v46 = v44[1];
    v47 = *v43;
    v67 = *v44;
    v71 = v43[1];
    v74 = *(_BYTE *)(sub_D95540(v71) + 8) == 14;
    if ( v74 == (*(_BYTE *)(sub_D95540(v46) + 8) == 14) )
    {
      if ( v67 == v47 )
      {
        if ( sub_D969D0(v71) )
        {
          sub_D969D0(v46);
        }
        else if ( sub_D969D0(v46) )
        {
LABEL_38:
          v45 = *v43;
          v11 += 2;
          v43 += 2;
          *(v11 - 2) = v45;
          *(v11 - 1) = *(v43 - 1);
          if ( v44 == v59 )
            break;
          continue;
        }
      }
      else if ( v47 != sub_F79730(v47, v67, a8) )
      {
        goto LABEL_38;
      }
    }
    else if ( *(_BYTE *)(sub_D95540(v71) + 8) == 14 )
    {
      goto LABEL_38;
    }
    v48 = *v44;
    v11 += 2;
    v44 += 2;
    *(v11 - 2) = v48;
    *(v11 - 1) = *(v44 - 1);
    if ( v44 == v59 )
      break;
  }
  while ( (__int64 *)a3 != v43 );
  v10 = v44;
LABEL_45:
  result = (__int64)v59;
  if ( v59 != v10 )
  {
    result = ((char *)v59 - (char *)v10) >> 4;
    if ( (char *)v59 - (char *)v10 > 0 )
    {
      do
      {
        v49 = *v10;
        v11 += 2;
        v10 += 2;
        *(v11 - 2) = v49;
        *(v11 - 1) = *(v10 - 1);
        --result;
      }
      while ( result );
    }
  }
  return result;
}
