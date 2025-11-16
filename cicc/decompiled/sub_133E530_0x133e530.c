// Function: sub_133E530
// Address: 0x133e530
//
_QWORD *__fastcall sub_133E530(_QWORD *a1)
{
  _QWORD *v1; // r8
  _QWORD *v3; // rbx
  _QWORD *v4; // rax
  _QWORD *v5; // r12
  _QWORD *v6; // rsi
  _QWORD *v7; // rcx
  unsigned __int64 v8; // rdi
  int v9; // edx
  __int64 v10; // rdx
  _QWORD *v11; // r11
  __int64 v12; // rdx
  _QWORD *v13; // rax
  _QWORD *v14; // rcx
  _QWORD *v15; // r9
  _QWORD *v16; // rdi
  unsigned __int64 v17; // r13
  int v18; // edx
  __int64 v19; // rdx
  _QWORD *v20; // rdx
  __int64 v21; // rdi
  _QWORD *v22; // rsi
  unsigned __int64 v23; // r11
  int v24; // edi
  __int64 v25; // rdi
  _QWORD *v26; // rax
  unsigned __int64 v27; // rcx
  int v28; // edx
  __int64 v29; // rdx
  _QWORD *v30; // r9
  _QWORD *v31; // rax
  _QWORD *v32; // rdx
  _QWORD *v33; // rsi
  _QWORD *v34; // r12
  unsigned __int64 v35; // rdi
  int v36; // ecx
  __int64 v37; // rcx
  _QWORD *v38; // rbx
  __int64 v39; // rdx
  _QWORD *v40; // rax
  _QWORD *v41; // rcx
  _QWORD *v42; // r11
  _QWORD *v43; // rdi
  unsigned __int64 v44; // r13
  int v45; // edx
  __int64 v46; // rdx
  _QWORD *v47; // rdx
  __int64 v48; // rdi
  _QWORD *v49; // rsi
  unsigned __int64 v50; // rbx
  int v51; // edi
  __int64 v52; // rdi
  __int64 v54; // rcx
  __int64 v55; // rdx
  __int64 v56; // rdx

  v1 = (_QWORD *)*a1;
  if ( *a1 )
  {
    v3 = v1 + 5;
    a1[1] = 0;
    v4 = (_QWORD *)v1[6];
    if ( !v4 )
      goto LABEL_50;
    v1[5] = 0;
    v3 = v4 + 5;
    v5 = v4;
    *(_QWORD *)(*a1 + 48LL) = 0;
    v4[5] = 0;
    v1 = (_QWORD *)v4[6];
    if ( !v1 )
    {
      v1 = v4;
LABEL_42:
      v26 = (_QWORD *)*a1;
      if ( *a1 )
      {
        v27 = v26[2] & 0xFFFLL;
        v28 = (v27 > (v1[2] & 0xFFFuLL)) - (v27 < (v1[2] & 0xFFFuLL));
        if ( v27 > (v1[2] & 0xFFFuLL) == v27 < (v1[2] & 0xFFFuLL) )
          v28 = (v26 > v5) - (v26 < v5);
        if ( v28 == -1 )
        {
          *v3 = v26;
          v55 = v26[7];
          v3[1] = v55;
          if ( v55 )
            *(_QWORD *)(v55 + 40) = v1;
          v26[7] = v1;
          v1 = v26;
          v3 = v26 + 5;
          *a1 = v26;
          goto LABEL_50;
        }
        v26[5] = v1;
        v29 = v3[2];
        v26[6] = v29;
        if ( v29 )
          *(_QWORD *)(v29 + 40) = v26;
        v3[2] = v26;
      }
      *a1 = v1;
LABEL_50:
      v30 = (_QWORD *)v3[2];
      if ( !v30 )
        goto LABEL_89;
      v31 = (_QWORD *)v30[6];
      v32 = v30 + 5;
      if ( !v31 )
        goto LABEL_89;
      v33 = (_QWORD *)v31[6];
      v34 = v31 + 5;
      if ( v33 )
        v33[5] = 0;
      *v32 = 0;
      v30[6] = 0;
      v31[5] = 0;
      v31[6] = 0;
      v35 = v30[2] & 0xFFFLL;
      v36 = (v35 > (v31[2] & 0xFFFuLL)) - (v35 < (v31[2] & 0xFFFuLL));
      if ( v35 > (v31[2] & 0xFFFuLL) == v35 < (v31[2] & 0xFFFuLL) )
        v36 = (v30 > v31) - (v30 < v31);
      if ( v36 == -1 )
      {
        v31[5] = v30;
        v54 = v30[7];
        v31[6] = v54;
        if ( v54 )
          *(_QWORD *)(v54 + 40) = v31;
        v30[7] = v31;
        v34 = v30 + 5;
      }
      else
      {
        *v32 = v31;
        v37 = v31[7];
        v30[6] = v37;
        if ( v37 )
          *(_QWORD *)(v37 + 40) = v30;
        v31[7] = v30;
        v30 = v31;
      }
      if ( v33 )
      {
        v38 = v30;
        while ( 1 )
        {
          v40 = (_QWORD *)v33[6];
          v41 = v33 + 5;
          if ( !v40 )
            break;
          v42 = (_QWORD *)v40[6];
          v43 = v40 + 5;
          if ( v42 )
            v42[5] = 0;
          *v41 = 0;
          v33[6] = 0;
          *v43 = 0;
          v40[6] = 0;
          v44 = v33[2] & 0xFFFLL;
          v45 = (v44 > (v40[2] & 0xFFFuLL)) - (v44 < (v40[2] & 0xFFFuLL));
          if ( v44 > (v40[2] & 0xFFFuLL) == v44 < (v40[2] & 0xFFFuLL) )
            v45 = (v33 > v40) - (v33 < v40);
          if ( v45 == -1 )
          {
            *v43 = v33;
            v46 = v33[7];
            v40[6] = v46;
            if ( v46 )
              *(_QWORD *)(v46 + 40) = v40;
            v33[7] = v40;
            v40 = v33;
            v38[6] = v33;
            if ( !v42 )
              goto LABEL_75;
          }
          else
          {
            *v41 = v40;
            v39 = v40[7];
            v33[6] = v39;
            if ( v39 )
              *(_QWORD *)(v39 + 40) = v33;
            v40[7] = v33;
            v38[6] = v40;
            if ( !v42 )
              goto LABEL_75;
          }
          v38 = v40;
          v33 = v42;
        }
        v38[6] = v33;
        v40 = v33;
      }
      else
      {
        v40 = v30;
      }
LABEL_75:
      v47 = (_QWORD *)v34[1];
      if ( !v47 )
      {
LABEL_89:
        *a1 = v30;
        return v1;
      }
      while ( 1 )
      {
        v49 = (_QWORD *)v47[6];
        v30[6] = 0;
        v47[6] = 0;
        if ( !v47 )
          goto LABEL_80;
        v50 = v30[2] & 0xFFFLL;
        v51 = (v50 > (v47[2] & 0xFFFuLL)) - (v50 < (v47[2] & 0xFFFuLL));
        if ( v50 > (v47[2] & 0xFFFuLL) == v50 < (v47[2] & 0xFFFuLL) )
          v51 = (v47 < v30) - (v47 > v30);
        if ( v51 != -1 )
          break;
        v47[5] = v30;
        v52 = v30[7];
        v47[6] = v52;
        if ( v52 )
          *(_QWORD *)(v52 + 40) = v47;
        v30[7] = v47;
        if ( !v49 )
          goto LABEL_89;
LABEL_81:
        v40[6] = v30;
        v47 = (_QWORD *)v49[6];
        v40 = v30;
        v30 = v49;
      }
      v30[5] = v47;
      v48 = v47[7];
      v30[6] = v48;
      if ( v48 )
        *(_QWORD *)(v48 + 40) = v30;
      v47[7] = v30;
      v30 = v47;
LABEL_80:
      if ( !v49 )
        goto LABEL_89;
      goto LABEL_81;
    }
    v6 = (_QWORD *)v1[6];
    v7 = v1 + 5;
    if ( v6 )
      v6[5] = 0;
    v4[5] = 0;
    v4[6] = 0;
    *v7 = 0;
    v1[6] = 0;
    v8 = v4[2] & 0xFFFLL;
    v9 = (v8 > (v1[2] & 0xFFFuLL)) - (v8 < (v1[2] & 0xFFFuLL));
    if ( v8 > (v1[2] & 0xFFFuLL) == v8 < (v1[2] & 0xFFFuLL) )
      v9 = (v4 > v1) - (v4 < v1);
    if ( v9 == -1 )
    {
      *v7 = v4;
      v56 = v4[7];
      v1[6] = v56;
      if ( v56 )
        *(_QWORD *)(v56 + 40) = v1;
      v4[7] = v1;
      v1 = v4;
    }
    else
    {
      v4[5] = v1;
      v10 = v1[7];
      v4[6] = v10;
      if ( v10 )
        *(_QWORD *)(v10 + 40) = v4;
      v1[7] = v4;
      v3 = v1 + 5;
      v5 = v1;
    }
    if ( v6 )
    {
      v11 = v1;
      while ( 1 )
      {
        v13 = (_QWORD *)v6[6];
        v14 = v6 + 5;
        if ( !v13 )
          break;
        v15 = (_QWORD *)v13[6];
        v16 = v13 + 5;
        if ( v15 )
          v15[5] = 0;
        *v14 = 0;
        v6[6] = 0;
        *v16 = 0;
        v13[6] = 0;
        v17 = v6[2] & 0xFFFLL;
        v18 = (v17 > (v13[2] & 0xFFFuLL)) - (v17 < (v13[2] & 0xFFFuLL));
        if ( v17 > (v13[2] & 0xFFFuLL) == v17 < (v13[2] & 0xFFFuLL) )
          v18 = (v6 > v13) - (v6 < v13);
        if ( v18 == -1 )
        {
          *v16 = v6;
          v19 = v6[7];
          v13[6] = v19;
          if ( v19 )
            *(_QWORD *)(v19 + 40) = v13;
          v6[7] = v13;
          v13 = v6;
          v11[6] = v6;
          if ( !v15 )
            goto LABEL_27;
        }
        else
        {
          *v14 = v13;
          v12 = v13[7];
          v6[6] = v12;
          if ( v12 )
            *(_QWORD *)(v12 + 40) = v6;
          v13[7] = v6;
          v11[6] = v13;
          if ( !v15 )
            goto LABEL_27;
        }
        v11 = v13;
        v6 = v15;
      }
      v11[6] = v6;
      v13 = v6;
    }
    else
    {
      v13 = v1;
    }
LABEL_27:
    v20 = (_QWORD *)v3[1];
    if ( !v20 )
      goto LABEL_42;
    while ( 1 )
    {
      v22 = (_QWORD *)v20[6];
      v1[6] = 0;
      v20[6] = 0;
      if ( !v20 )
        goto LABEL_32;
      v23 = v1[2] & 0xFFFLL;
      v24 = (v23 > (v20[2] & 0xFFFuLL)) - (v23 < (v20[2] & 0xFFFuLL));
      if ( v23 > (v20[2] & 0xFFFuLL) == v23 < (v20[2] & 0xFFFuLL) )
        v24 = (v20 < v1) - (v20 > v1);
      if ( v24 != -1 )
        break;
      v20[5] = v1;
      v25 = v1[7];
      v20[6] = v25;
      if ( v25 )
        *(_QWORD *)(v25 + 40) = v20;
      v1[7] = v20;
      if ( !v22 )
      {
LABEL_41:
        v5 = v1;
        v3 = v1 + 5;
        goto LABEL_42;
      }
LABEL_33:
      v13[6] = v1;
      v20 = (_QWORD *)v22[6];
      v13 = v1;
      v1 = v22;
    }
    v1[5] = v20;
    v21 = v20[7];
    v1[6] = v21;
    if ( v21 )
      *(_QWORD *)(v21 + 40) = v1;
    v20[7] = v1;
    v1 = v20;
LABEL_32:
    if ( !v22 )
      goto LABEL_41;
    goto LABEL_33;
  }
  return 0;
}
