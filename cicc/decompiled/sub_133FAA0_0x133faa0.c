// Function: sub_133FAA0
// Address: 0x133faa0
//
_QWORD *__fastcall sub_133FAA0(_QWORD *a1)
{
  _QWORD *v1; // r8
  _QWORD *v3; // rbx
  _QWORD *v4; // rax
  _QWORD *v5; // rcx
  _QWORD *v6; // rsi
  int v7; // edx
  __int64 v8; // rdx
  _QWORD *v9; // r11
  __int64 v10; // rdx
  _QWORD *v11; // rax
  _QWORD *v12; // rsi
  _QWORD *v13; // r9
  _QWORD *v14; // rdi
  int v15; // edx
  __int64 v16; // rdx
  _QWORD *v17; // rdx
  __int64 v18; // rdi
  _QWORD *v19; // rsi
  int v20; // edi
  __int64 v21; // rdi
  _QWORD *v22; // rax
  unsigned __int64 v23; // rdi
  int v24; // edx
  __int64 v25; // rdx
  _QWORD *v26; // r9
  _QWORD *v27; // rax
  _QWORD *v28; // rdx
  _QWORD *v29; // rcx
  _QWORD *v30; // r12
  unsigned __int64 v31; // rbx
  int v32; // esi
  __int64 v33; // rsi
  _QWORD *v34; // rbx
  __int64 v35; // rdx
  _QWORD *v36; // rax
  _QWORD *v37; // rsi
  _QWORD *v38; // r11
  _QWORD *v39; // rdi
  int v40; // edx
  __int64 v41; // rdx
  _QWORD *v42; // rdx
  __int64 v43; // rdi
  _QWORD *v44; // rsi
  int v45; // edi
  __int64 v46; // rdi
  __int64 v48; // rsi
  __int64 v49; // rdx
  __int64 v50; // rdx

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
    *(_QWORD *)(*a1 + 48LL) = 0;
    v4[5] = 0;
    v1 = (_QWORD *)v4[6];
    if ( !v1 )
    {
      v1 = v4;
LABEL_42:
      v22 = (_QWORD *)*a1;
      if ( *a1 )
      {
        v23 = v1[4];
        v24 = (v22[4] > v23) - (v22[4] < v23);
        if ( v22[4] > v23 == v22[4] < v23 )
          v24 = (v22[1] > v1[1]) - (v22[1] < v1[1]);
        if ( v24 == -1 )
        {
          *v3 = v22;
          v49 = v22[7];
          v3[1] = v49;
          if ( v49 )
            *(_QWORD *)(v49 + 40) = v1;
          v22[7] = v1;
          v1 = v22;
          v3 = v22 + 5;
          *a1 = v22;
          goto LABEL_50;
        }
        v22[5] = v1;
        v25 = v3[2];
        v22[6] = v25;
        if ( v25 )
          *(_QWORD *)(v25 + 40) = v22;
        v3[2] = v22;
      }
      *a1 = v1;
LABEL_50:
      v26 = (_QWORD *)v3[2];
      if ( !v26 )
        goto LABEL_89;
      v27 = (_QWORD *)v26[6];
      v28 = v26 + 5;
      if ( !v27 )
        goto LABEL_89;
      v29 = (_QWORD *)v27[6];
      v30 = v27 + 5;
      if ( v29 )
        v29[5] = 0;
      *v28 = 0;
      v26[6] = 0;
      v27[5] = 0;
      v27[6] = 0;
      v31 = v27[4];
      v32 = (v26[4] > v31) - (v26[4] < v31);
      if ( v26[4] > v31 == v26[4] < v31 )
        v32 = (v26[1] > v27[1]) - (v26[1] < v27[1]);
      if ( v32 == -1 )
      {
        v27[5] = v26;
        v48 = v26[7];
        v27[6] = v48;
        if ( v48 )
          *(_QWORD *)(v48 + 40) = v27;
        v26[7] = v27;
        v30 = v26 + 5;
      }
      else
      {
        *v28 = v27;
        v33 = v27[7];
        v26[6] = v33;
        if ( v33 )
          *(_QWORD *)(v33 + 40) = v26;
        v27[7] = v26;
        v26 = v27;
      }
      if ( v29 )
      {
        v34 = v26;
        while ( 1 )
        {
          v36 = (_QWORD *)v29[6];
          v37 = v29 + 5;
          if ( !v36 )
            break;
          v38 = (_QWORD *)v36[6];
          v39 = v36 + 5;
          if ( v38 )
            v38[5] = 0;
          *v37 = 0;
          v29[6] = 0;
          *v39 = 0;
          v36[6] = 0;
          v40 = (v29[4] > v36[4]) - (v29[4] < v36[4]);
          if ( v29[4] > v36[4] == v29[4] < v36[4] )
            v40 = (v29[1] > v36[1]) - (v29[1] < v36[1]);
          if ( v40 == -1 )
          {
            *v39 = v29;
            v41 = v29[7];
            v36[6] = v41;
            if ( v41 )
              *(_QWORD *)(v41 + 40) = v36;
            v29[7] = v36;
            v36 = v29;
            v34[6] = v29;
            if ( !v38 )
              goto LABEL_75;
          }
          else
          {
            *v37 = v36;
            v35 = v36[7];
            v29[6] = v35;
            if ( v35 )
              *(_QWORD *)(v35 + 40) = v29;
            v36[7] = v29;
            v34[6] = v36;
            if ( !v38 )
              goto LABEL_75;
          }
          v34 = v36;
          v29 = v38;
        }
        v34[6] = v29;
        v36 = v29;
      }
      else
      {
        v36 = v26;
      }
LABEL_75:
      v42 = (_QWORD *)v30[1];
      if ( !v42 )
      {
LABEL_89:
        *a1 = v26;
        return v1;
      }
      while ( 1 )
      {
        v44 = (_QWORD *)v42[6];
        v26[6] = 0;
        v42[6] = 0;
        if ( !v42 )
          goto LABEL_80;
        v45 = (v26[4] > v42[4]) - (v26[4] < v42[4]);
        if ( v26[4] > v42[4] == v26[4] < v42[4] )
          v45 = (v26[1] > v42[1]) - (v26[1] < v42[1]);
        if ( v45 != -1 )
          break;
        v42[5] = v26;
        v46 = v26[7];
        v42[6] = v46;
        if ( v46 )
          *(_QWORD *)(v46 + 40) = v42;
        v26[7] = v42;
        if ( !v44 )
          goto LABEL_89;
LABEL_81:
        v36[6] = v26;
        v42 = (_QWORD *)v44[6];
        v36 = v26;
        v26 = v44;
      }
      v26[5] = v42;
      v43 = v42[7];
      v26[6] = v43;
      if ( v43 )
        *(_QWORD *)(v43 + 40) = v26;
      v42[7] = v26;
      v26 = v42;
LABEL_80:
      if ( !v44 )
        goto LABEL_89;
      goto LABEL_81;
    }
    v5 = (_QWORD *)v1[6];
    v6 = v1 + 5;
    if ( v5 )
      v5[5] = 0;
    v4[5] = 0;
    v4[6] = 0;
    *v6 = 0;
    v1[6] = 0;
    v7 = (v4[4] > v1[4]) - (v4[4] < v1[4]);
    if ( v4[4] > v1[4] == v4[4] < v1[4] )
      v7 = (v4[1] > v1[1]) - (v4[1] < v1[1]);
    if ( v7 == -1 )
    {
      *v6 = v4;
      v50 = v4[7];
      v1[6] = v50;
      if ( v50 )
        *(_QWORD *)(v50 + 40) = v1;
      v4[7] = v1;
      v1 = v4;
    }
    else
    {
      v4[5] = v1;
      v8 = v1[7];
      v4[6] = v8;
      if ( v8 )
        *(_QWORD *)(v8 + 40) = v4;
      v1[7] = v4;
      v3 = v1 + 5;
    }
    if ( v5 )
    {
      v9 = v1;
      while ( 1 )
      {
        v11 = (_QWORD *)v5[6];
        v12 = v5 + 5;
        if ( !v11 )
          break;
        v13 = (_QWORD *)v11[6];
        v14 = v11 + 5;
        if ( v13 )
          v13[5] = 0;
        *v12 = 0;
        v5[6] = 0;
        *v14 = 0;
        v11[6] = 0;
        v15 = (v5[4] > v11[4]) - (v5[4] < v11[4]);
        if ( v5[4] > v11[4] == v5[4] < v11[4] )
          v15 = (v5[1] > v11[1]) - (v5[1] < v11[1]);
        if ( v15 == -1 )
        {
          *v14 = v5;
          v16 = v5[7];
          v11[6] = v16;
          if ( v16 )
            *(_QWORD *)(v16 + 40) = v11;
          v5[7] = v11;
          v11 = v5;
          v9[6] = v5;
          if ( !v13 )
            goto LABEL_27;
        }
        else
        {
          *v12 = v11;
          v10 = v11[7];
          v5[6] = v10;
          if ( v10 )
            *(_QWORD *)(v10 + 40) = v5;
          v11[7] = v5;
          v9[6] = v11;
          if ( !v13 )
            goto LABEL_27;
        }
        v9 = v11;
        v5 = v13;
      }
      v9[6] = v5;
      v11 = v5;
    }
    else
    {
      v11 = v1;
    }
LABEL_27:
    v17 = (_QWORD *)v3[1];
    if ( !v17 )
      goto LABEL_42;
    while ( 1 )
    {
      v19 = (_QWORD *)v17[6];
      v1[6] = 0;
      v17[6] = 0;
      if ( !v17 )
        goto LABEL_32;
      v20 = (v1[4] > v17[4]) - (v1[4] < v17[4]);
      if ( v1[4] > v17[4] == v1[4] < v17[4] )
        v20 = (v1[1] > v17[1]) - (v1[1] < v17[1]);
      if ( v20 != -1 )
        break;
      v17[5] = v1;
      v21 = v1[7];
      v17[6] = v21;
      if ( v21 )
        *(_QWORD *)(v21 + 40) = v17;
      v1[7] = v17;
      if ( !v19 )
      {
LABEL_41:
        v3 = v1 + 5;
        goto LABEL_42;
      }
LABEL_33:
      v11[6] = v1;
      v17 = (_QWORD *)v19[6];
      v11 = v1;
      v1 = v19;
    }
    v1[5] = v17;
    v18 = v17[7];
    v1[6] = v18;
    if ( v18 )
      *(_QWORD *)(v18 + 40) = v1;
    v17[7] = v1;
    v1 = v17;
LABEL_32:
    if ( !v19 )
      goto LABEL_41;
    goto LABEL_33;
  }
  return 0;
}
