// Function: sub_E1D370
// Address: 0xe1d370
//
__int64 __fastcall sub_E1D370(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v6; // r12
  unsigned __int8 *v7; // rdx
  char *v8; // rax
  __int64 result; // rax
  __int64 v10; // r8
  __int64 v11; // r9
  char *v12; // rdx
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // r13
  __int64 v18; // r12
  char v19; // dl
  __int64 v20; // r13
  char *v21; // rax
  char *v22; // rsi
  char *v23; // rax
  __int64 v24; // rsi
  __int64 v25; // r12
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // r8
  __int64 v29; // r9
  __int64 v30; // rax
  char v31; // dl
  int v32; // eax
  char *v33; // rax
  unsigned __int8 *v34; // rax
  __int64 v35; // rcx
  __int64 v36; // rdx
  unsigned __int64 v37; // rdx
  __int64 v38; // rcx
  __int64 v39; // rax
  __int64 v40; // rdx
  __int64 v41; // rcx
  __int64 v42; // r8
  __int64 v43; // r9
  __int64 v44; // rdx
  __int64 v45; // rcx
  __int64 v46; // r8
  __int64 v47; // r9
  __int64 v48; // rsi
  __int64 v49; // rdx
  char *v50; // rcx
  __int64 v51; // rdx
  __int64 v52; // rcx
  __int64 v53; // r8
  __int64 v54; // r9
  char *v55; // rax
  __int64 v56; // rdx
  __int64 v57; // rcx
  __int64 v58; // r8
  __int64 v59; // r9
  __int64 v60; // [rsp+8h] [rbp-D8h]
  char v61; // [rsp+1Fh] [rbp-C1h] BYREF
  __int64 v62[24]; // [rsp+20h] [rbp-C0h] BYREF

  v6 = a2;
  v7 = *(unsigned __int8 **)(a1 + 8);
  v8 = *(char **)a1;
  if ( *(unsigned __int8 **)a1 == v7 )
    goto LABEL_4;
  if ( *v8 == 78 )
  {
    *(_QWORD *)a1 = v8 + 1;
    if ( v7 != (unsigned __int8 *)(v8 + 1) && v8[1] == 72 )
    {
      v34 = (unsigned __int8 *)(v8 + 2);
      *(_QWORD *)a1 = v34;
      if ( a2 )
      {
        *((_BYTE *)a2 + 24) = 1;
        v34 = *(unsigned __int8 **)a1;
        v7 = *(unsigned __int8 **)(a1 + 8);
      }
      goto LABEL_35;
    }
    v32 = sub_E0E0E0(a1);
    if ( !a2 )
    {
      v34 = *(unsigned __int8 **)(a1 + 8);
      v50 = *(char **)a1;
      v7 = v34;
      if ( *(unsigned __int8 **)a1 != v34 )
      {
        if ( *v50 == 79 || *v50 == 82 )
        {
          v34 = (unsigned __int8 *)(v50 + 1);
          *(_QWORD *)a1 = v50 + 1;
        }
        else
        {
          v34 = *(unsigned __int8 **)a1;
        }
      }
      goto LABEL_35;
    }
    *((_DWORD *)a2 + 1) = v32;
    v33 = *(char **)a1;
    if ( *(_QWORD *)a1 != *(_QWORD *)(a1 + 8) )
    {
      if ( *v33 == 79 )
      {
        *(_QWORD *)a1 = v33 + 1;
        *((_BYTE *)a2 + 8) = 2;
        v34 = *(unsigned __int8 **)a1;
        v7 = *(unsigned __int8 **)(a1 + 8);
        goto LABEL_35;
      }
      if ( *v33 == 82 )
      {
        *(_QWORD *)a1 = v33 + 1;
        *((_BYTE *)a2 + 8) = 1;
        v34 = *(unsigned __int8 **)a1;
        v7 = *(unsigned __int8 **)(a1 + 8);
        goto LABEL_35;
      }
    }
    *((_BYTE *)a2 + 8) = 0;
    v34 = *(unsigned __int8 **)a1;
    v7 = *(unsigned __int8 **)(a1 + 8);
LABEL_35:
    v62[0] = 0;
    while ( 1 )
    {
      if ( v7 == v34 )
      {
        if ( !v6 )
          goto LABEL_46;
      }
      else
      {
        v35 = *v34;
        if ( (_BYTE)v35 == 69 )
        {
          *(_QWORD *)a1 = v34 + 1;
          result = v62[0];
          if ( v62[0] )
          {
            v49 = *(_QWORD *)(a1 + 304);
            if ( *(_QWORD *)(a1 + 296) != v49 )
            {
              *(_QWORD *)(a1 + 304) = v49 - 8;
              return result;
            }
          }
          return 0;
        }
        if ( !v6 )
        {
          v37 = v7 - v34;
          goto LABEL_41;
        }
      }
      *((_BYTE *)v6 + 1) = 0;
      v36 = *(_QWORD *)(a1 + 8);
      v34 = *(unsigned __int8 **)a1;
      if ( v36 == *(_QWORD *)a1 )
        goto LABEL_46;
      v35 = *v34;
      v37 = v36 - (_QWORD)v34;
LABEL_41:
      if ( (_BYTE)v35 == 84 )
      {
        if ( v62[0] )
          return 0;
        v39 = sub_E18810(a1, (__int64)a2, v37, v35, a5);
        v62[0] = v39;
        goto LABEL_48;
      }
      if ( (_BYTE)v35 == 73 )
      {
        if ( !v62[0] )
          return 0;
        v44 = sub_E1F700(a1, v6 != 0);
        if ( !v44 )
          return 0;
        v48 = v62[0];
        if ( *(_BYTE *)(v62[0] + 8) == 45 )
          return 0;
        if ( v6 )
          *((_BYTE *)v6 + 1) = 1;
        v39 = sub_E0FC10(a1 + 816, v48, v44, v45, v46, v47);
        v62[0] = v39;
        goto LABEL_48;
      }
      if ( v37 > 1 && (_BYTE)v35 == 68 )
      {
        if ( (v34[1] & 0xDF) != 0x54 )
        {
LABEL_46:
          v38 = 0;
LABEL_47:
          v39 = sub_E1E740(a1, v6, v62[0], v38);
          v62[0] = v39;
          goto LABEL_48;
        }
        if ( v62[0] )
          return 0;
        v39 = sub_E1AB20((_QWORD *)a1);
        v62[0] = v39;
LABEL_48:
        if ( !v39 )
          return 0;
        a2 = v62;
        sub_E18380(a1 + 296, v62, v40, v41, v42, v43);
        v34 = *(unsigned __int8 **)a1;
        v7 = *(unsigned __int8 **)(a1 + 8);
        if ( *(unsigned __int8 **)a1 != v7 && *v34 == 77 )
          *(_QWORD *)a1 = ++v34;
      }
      else
      {
        if ( (_BYTE)v35 != 83 )
          goto LABEL_46;
        if ( v37 > 1 && v34[1] == 116 )
        {
          a2 = (__int64 *)"std";
          *(_QWORD *)a1 = v34 + 2;
          v38 = sub_E0FD70(a1 + 816, "std");
        }
        else
        {
          v38 = sub_E18570(a1, (__int64)a2, v37, v35, a5, a6);
        }
        if ( !v38 )
          return 0;
        if ( *(_BYTE *)(v38 + 8) == 27 )
          goto LABEL_47;
        if ( v62[0] )
          return 0;
        v34 = *(unsigned __int8 **)a1;
        v7 = *(unsigned __int8 **)(a1 + 8);
        v62[0] = v38;
      }
    }
  }
  if ( *v8 == 90 )
  {
    *(_QWORD *)a1 = v8 + 1;
    v20 = sub_E1C560((const void **)a1, 1);
    if ( !v20 )
      return 0;
    v21 = *(char **)a1;
    v22 = *(char **)(a1 + 8);
    if ( *(char **)a1 == v22 || *v21 != 69 )
      return 0;
    *(_QWORD *)a1 = v21 + 1;
    if ( v22 != v21 + 1 && v21[1] == 115 )
    {
      *(_QWORD *)a1 = v21 + 2;
      *(_QWORD *)a1 = sub_E182C0(v21 + 2, v22);
      v51 = sub_E0FD70(a1 + 816, "string literal");
      if ( !v51 )
        return 0;
      return sub_E0FCC0(a1 + 816, v20, v51, v52, v53, v54);
    }
    sub_E0F2D0(v62, (_QWORD *)a1);
    v23 = *(char **)a1;
    if ( *(_QWORD *)a1 != *(_QWORD *)(a1 + 8) && *v23 == 100 )
    {
      v24 = 1;
      *(_QWORD *)a1 = v23 + 1;
      sub_E0DEF0((char **)a1, 1);
      v55 = *(char **)a1;
      if ( *(_QWORD *)a1 != *(_QWORD *)(a1 + 8) && *v55 == 95 )
      {
        v24 = (__int64)v6;
        *(_QWORD *)a1 = v55 + 1;
        v56 = sub_E1D370(a1, v6);
        if ( v56 )
        {
          v24 = v20;
          v30 = sub_E0FCC0(a1 + 816, v20, v56, v57, v58, v59);
          goto LABEL_26;
        }
      }
    }
    else
    {
      v24 = (__int64)v6;
      v25 = sub_E1D370(a1, v6);
      if ( v25 )
      {
        v24 = 32;
        *(_QWORD *)a1 = sub_E182C0(*(char **)a1, *(char **)(a1 + 8));
        v30 = sub_E0E790(a1 + 816, 32, v26, v27, v28, v29);
        if ( v30 )
        {
          v31 = *(_BYTE *)(v30 + 10);
          *(_WORD *)(v30 + 8) = 16410;
          *(_QWORD *)(v30 + 16) = v20;
          *(_QWORD *)(v30 + 24) = v25;
          *(_BYTE *)(v30 + 10) = v31 & 0xF0 | 5;
          *(_QWORD *)v30 = &unk_49DF848;
        }
        goto LABEL_26;
      }
    }
    v30 = 0;
LABEL_26:
    v60 = v30;
    sub_E0F090(v62, (const void *)v24);
    return v60;
  }
LABEL_4:
  v62[0] = 0;
  v61 = 0;
  result = sub_E1EC70(a1, a2, &v61);
  v62[0] = result;
  if ( !result )
    return 0;
  v12 = *(char **)a1;
  if ( *(_QWORD *)(a1 + 8) != *(_QWORD *)a1 && *v12 == 73 )
  {
    if ( !v61 )
      sub_E18380(a1 + 296, v62, (__int64)v12, 0, v10, v11);
    v17 = sub_E1F700(a1, a2 != 0);
    if ( !v17 )
      return 0;
    if ( a2 )
      *((_BYTE *)a2 + 1) = 1;
    v18 = v62[0];
    result = sub_E0E790(a1 + 816, 32, v13, v14, v15, v16);
    if ( result )
    {
      *(_QWORD *)(result + 16) = v18;
      *(_WORD *)(result + 8) = 16429;
      v19 = *(_BYTE *)(result + 10);
      *(_QWORD *)(result + 24) = v17;
      *(_BYTE *)(result + 10) = v19 & 0xF0 | 5;
      *(_QWORD *)result = &unk_49DFEA8;
    }
  }
  else if ( v61 )
  {
    return 0;
  }
  return result;
}
