// Function: sub_1348F60
// Address: 0x1348f60
//
_QWORD *__fastcall sub_1348F60(_QWORD *a1, _QWORD *a2)
{
  _QWORD *v3; // r12
  _QWORD *v4; // rbx
  __int64 v5; // r11
  _QWORD *v6; // rax
  __int64 v7; // rcx
  _QWORD *v8; // rdi
  _QWORD *v9; // rdx
  _QWORD *v10; // rsi
  _QWORD *v11; // r12
  __int64 v12; // rcx
  _QWORD *v13; // r13
  __int64 v14; // r9
  _QWORD *v15; // rax
  _QWORD *v16; // rdx
  _QWORD *v17; // r8
  _QWORD *v18; // rcx
  __int64 v19; // r9
  _QWORD *v20; // rdx
  __int64 v21; // r9
  _QWORD *v22; // rsi
  __int64 v23; // r9
  _QWORD *result; // rax
  _QWORD *v26; // rsi
  _QWORD *v27; // rdx
  _QWORD *v28; // rax
  _QWORD *v29; // rdi
  _QWORD *v30; // r14
  __int64 v31; // rcx
  _QWORD *v32; // r13
  __int64 v33; // r11
  _QWORD *v34; // rax
  _QWORD *v35; // rdx
  _QWORD *v36; // r9
  _QWORD *v37; // rcx
  __int64 v38; // r11
  __int64 v39; // rcx
  __int64 v40; // rax
  _QWORD *v41; // rax
  __int64 v42; // rdx
  _QWORD *v43; // rdi
  _QWORD *v44; // rdx
  _QWORD *v45; // rsi
  _QWORD *v46; // rbx
  __int64 v47; // rcx
  _QWORD *v48; // r11
  __int64 v49; // r10
  _QWORD *v50; // rdx
  _QWORD *v51; // r9
  _QWORD *v52; // rcx
  __int64 v53; // r10
  _QWORD *v54; // rdx
  __int64 v55; // r11
  _QWORD *v56; // rdi
  __int64 v57; // r11
  __int64 v58; // rdx
  __int64 v59; // rcx
  _QWORD *v60; // rdx
  __int64 v61; // r10
  _QWORD *v62; // rsi
  __int64 v63; // r10
  __int64 v64; // rcx

  v3 = (_QWORD *)*a1;
  v4 = a2 + 5;
  if ( a2 == (_QWORD *)*a1 )
  {
    result = a2 + 5;
    if ( !a2[7] )
    {
      result = (_QWORD *)a2[6];
      *a1 = result;
      if ( result )
        result[5] = 0;
      return result;
    }
    a1[1] = 0;
    v26 = (_QWORD *)a2[6];
    if ( v26 )
    {
      a2[5] = 0;
      v27 = v26 + 5;
      *(_QWORD *)(*a1 + 48LL) = 0;
      v26[5] = 0;
      v28 = (_QWORD *)v26[6];
      if ( v28 )
      {
        v29 = (_QWORD *)v28[6];
        v30 = v28 + 5;
        if ( v29 )
          v29[5] = 0;
        *v27 = 0;
        v26[6] = 0;
        v28[5] = 0;
        v28[6] = 0;
        if ( (v26[1] > v28[1]) - (v26[1] < v28[1]) == -1 )
        {
          v28[5] = v26;
          v59 = v26[7];
          v28[6] = v59;
          if ( v59 )
            *(_QWORD *)(v59 + 40) = v28;
          v26[7] = v28;
          v30 = v26 + 5;
        }
        else
        {
          *v27 = v28;
          v31 = v28[7];
          v26[6] = v31;
          if ( v31 )
            *(_QWORD *)(v31 + 40) = v26;
          v28[7] = v26;
          v26 = v28;
        }
        if ( v29 )
        {
          v32 = v26;
          while ( 1 )
          {
            v34 = (_QWORD *)v29[6];
            v35 = v29 + 5;
            if ( !v34 )
              break;
            v36 = (_QWORD *)v34[6];
            v37 = v34 + 5;
            if ( v36 )
              v36[5] = 0;
            *v35 = 0;
            v29[6] = 0;
            *v37 = 0;
            v34[6] = 0;
            if ( (v29[1] > v34[1]) - (v29[1] < v34[1]) == -1 )
            {
              *v37 = v29;
              v38 = v29[7];
              v34[6] = v38;
              if ( v38 )
                *(_QWORD *)(v38 + 40) = v34;
              v29[7] = v34;
              v34 = v29;
            }
            else
            {
              *v35 = v34;
              v33 = v34[7];
              v29[6] = v33;
              if ( v33 )
                *(_QWORD *)(v33 + 40) = v29;
              v34[7] = v29;
            }
            v32[6] = v34;
            if ( !v36 )
              goto LABEL_113;
            v32 = v34;
            v29 = v36;
          }
          v32[6] = v29;
          v34 = v29;
        }
        else
        {
          v34 = v26;
        }
LABEL_113:
        v54 = (_QWORD *)v30[1];
        if ( v54 )
        {
          while ( 1 )
          {
            v56 = (_QWORD *)v54[6];
            v26[6] = 0;
            v54[6] = 0;
            if ( v54 )
            {
              if ( (v26[1] > v54[1]) - (v26[1] < v54[1]) == -1 )
              {
                v54[5] = v26;
                v57 = v26[7];
                v54[6] = v57;
                if ( v57 )
                  *(_QWORD *)(v57 + 40) = v54;
                v26[7] = v54;
              }
              else
              {
                v26[5] = v54;
                v55 = v54[7];
                v26[6] = v55;
                if ( v55 )
                  *(_QWORD *)(v55 + 40) = v26;
                v54[7] = v26;
                v26 = v54;
              }
            }
            if ( !v56 )
              break;
            v34[6] = v26;
            v54 = (_QWORD *)v56[6];
            v34 = v26;
            v26 = v56;
          }
        }
      }
      v41 = (_QWORD *)*a1;
      if ( *a1 )
      {
        if ( (v41[1] > v26[1]) - (v41[1] < v26[1]) == -1 )
        {
          v26[5] = v41;
          v58 = v41[7];
          v26[6] = v58;
          if ( v58 )
            *(_QWORD *)(v58 + 40) = v26;
          v41[7] = v26;
          v26 = v41;
        }
        else
        {
          v41[5] = v26;
          v42 = v26[7];
          v41[6] = v42;
          if ( v42 )
            *(_QWORD *)(v42 + 40) = v41;
          v26[7] = v41;
        }
      }
      *a1 = v26;
      if ( v3 != v26 )
        goto LABEL_2;
      result = v3 + 5;
    }
    v43 = (_QWORD *)result[2];
    if ( v43 )
    {
      result = (_QWORD *)v43[6];
      v44 = v43 + 5;
      if ( result )
      {
        v45 = (_QWORD *)result[6];
        v46 = result + 5;
        if ( v45 )
          v45[5] = 0;
        *v44 = 0;
        v43[6] = 0;
        result[5] = 0;
        result[6] = 0;
        if ( (v43[1] > result[1]) - (v43[1] < result[1]) == -1 )
        {
          result[5] = v43;
          v64 = v43[7];
          result[6] = v64;
          if ( v64 )
            *(_QWORD *)(v64 + 40) = result;
          v43[7] = result;
          v46 = v43 + 5;
        }
        else
        {
          *v44 = result;
          v47 = result[7];
          v43[6] = v47;
          if ( v47 )
            *(_QWORD *)(v47 + 40) = v43;
          result[7] = v43;
          v43 = result;
        }
        if ( v45 )
        {
          v48 = v43;
          while ( 1 )
          {
            result = (_QWORD *)v45[6];
            v50 = v45 + 5;
            if ( !result )
              break;
            v51 = (_QWORD *)result[6];
            v52 = result + 5;
            if ( v51 )
              v51[5] = 0;
            *v50 = 0;
            v45[6] = 0;
            *v52 = 0;
            result[6] = 0;
            if ( (v45[1] > result[1]) - (v45[1] < result[1]) == -1 )
            {
              *v52 = v45;
              v53 = v45[7];
              result[6] = v53;
              if ( v53 )
                *(_QWORD *)(v53 + 40) = result;
              v45[7] = result;
              result = v45;
            }
            else
            {
              *v50 = result;
              v49 = result[7];
              v45[6] = v49;
              if ( v49 )
                *(_QWORD *)(v49 + 40) = v45;
              result[7] = v45;
            }
            v48[6] = result;
            if ( !v51 )
              goto LABEL_138;
            v48 = result;
            v45 = v51;
          }
          v48[6] = v45;
          result = v45;
        }
        else
        {
          result = v43;
        }
LABEL_138:
        v60 = (_QWORD *)v46[1];
        if ( v60 )
        {
          while ( 1 )
          {
            v62 = (_QWORD *)v60[6];
            v43[6] = 0;
            v60[6] = 0;
            if ( v60 )
            {
              if ( (v43[1] > v60[1]) - (v43[1] < v60[1]) == -1 )
              {
                v60[5] = v43;
                v63 = v43[7];
                v60[6] = v63;
                if ( v63 )
                  *(_QWORD *)(v63 + 40) = v60;
                v43[7] = v60;
              }
              else
              {
                v43[5] = v60;
                v61 = v60[7];
                v43[6] = v61;
                if ( v61 )
                  *(_QWORD *)(v61 + 40) = v43;
                v60[7] = v43;
                v43 = v60;
              }
            }
            if ( !v62 )
              break;
            result[6] = v43;
            v60 = (_QWORD *)v62[6];
            result = v43;
            v43 = v62;
          }
        }
      }
    }
    *a1 = v43;
    return result;
  }
LABEL_2:
  v5 = a2[5];
  v6 = (_QWORD *)v4[2];
  if ( v5 && (v7 = v5 + 40, a2 == *(_QWORD **)(v5 + 56)) )
  {
    if ( v6 )
    {
      v8 = (_QWORD *)v6[6];
      v9 = v6 + 5;
      if ( !v8 )
      {
        v11 = v6 + 5;
        v8 = (_QWORD *)v4[2];
        goto LABEL_41;
      }
      goto LABEL_7;
    }
    v40 = v4[1];
    *(_QWORD *)(v5 + 56) = v40;
    if ( v40 )
      *(_QWORD *)(v40 + 40) = v5;
  }
  else
  {
    if ( v6 )
    {
      v8 = (_QWORD *)v6[6];
      v9 = v6 + 5;
      if ( !v8 )
        goto LABEL_76;
      v5 = 0;
LABEL_7:
      v10 = (_QWORD *)v8[6];
      v11 = v8 + 5;
      if ( v10 )
        v10[5] = 0;
      *v9 = 0;
      v9[1] = 0;
      *v11 = 0;
      v8[6] = 0;
      if ( (v6[1] > v8[1]) - (v6[1] < v8[1]) == -1 )
      {
        *v11 = v6;
        v39 = v9[2];
        v8[6] = v39;
        if ( v39 )
          *(_QWORD *)(v39 + 40) = v8;
        v9[2] = v8;
        v11 = v9;
        v8 = v6;
      }
      else
      {
        *v9 = v8;
        v12 = v8[7];
        v9[1] = v12;
        if ( v12 )
          *(_QWORD *)(v12 + 40) = v6;
        v8[7] = v6;
      }
      if ( v10 )
      {
        v13 = v8;
        while ( 1 )
        {
          v15 = (_QWORD *)v10[6];
          v16 = v10 + 5;
          if ( !v15 )
            break;
          v17 = (_QWORD *)v15[6];
          v18 = v15 + 5;
          if ( v17 )
            v17[5] = 0;
          *v16 = 0;
          v10[6] = 0;
          *v18 = 0;
          v15[6] = 0;
          if ( (v10[1] > v15[1]) - (v10[1] < v15[1]) == -1 )
          {
            *v18 = v10;
            v19 = v10[7];
            v15[6] = v19;
            if ( v19 )
              *(_QWORD *)(v19 + 40) = v15;
            v10[7] = v15;
            v15 = v10;
            v13[6] = v10;
            if ( !v17 )
              goto LABEL_26;
          }
          else
          {
            *v16 = v15;
            v14 = v15[7];
            v10[6] = v14;
            if ( v14 )
              *(_QWORD *)(v14 + 40) = v10;
            v15[7] = v10;
            v13[6] = v15;
            if ( !v17 )
              goto LABEL_26;
          }
          v13 = v15;
          v10 = v17;
        }
        v13[6] = v10;
        v15 = v10;
      }
      else
      {
        v15 = v8;
      }
LABEL_26:
      v20 = (_QWORD *)v11[1];
      if ( !v20 )
      {
LABEL_39:
        if ( v5 )
        {
          v7 = v5 + 40;
LABEL_41:
          *v11 = v5;
          *(_QWORD *)(v7 + 16) = v8;
LABEL_42:
          result = (_QWORD *)v4[1];
          v11[1] = result;
          if ( result )
            result[5] = v8;
          return result;
        }
        v5 = a2[5];
        v9 = v11;
        v6 = v8;
LABEL_76:
        *v9 = v5;
        if ( v5 )
          *(_QWORD *)(v5 + 48) = v6;
        v11 = v9;
        v8 = v6;
        goto LABEL_42;
      }
      while ( 1 )
      {
        v22 = (_QWORD *)v20[6];
        v8[6] = 0;
        v20[6] = 0;
        if ( !v20 )
          goto LABEL_31;
        if ( (v8[1] > v20[1]) - (v8[1] < v20[1]) != -1 )
          break;
        v20[5] = v8;
        v23 = v8[7];
        v20[6] = v23;
        if ( v23 )
          *(_QWORD *)(v23 + 40) = v20;
        v8[7] = v20;
        if ( !v22 )
        {
LABEL_38:
          v11 = v8 + 5;
          goto LABEL_39;
        }
LABEL_32:
        v15[6] = v8;
        v20 = (_QWORD *)v22[6];
        v15 = v8;
        v8 = v22;
      }
      v8[5] = v20;
      v21 = v20[7];
      v8[6] = v21;
      if ( v21 )
        *(_QWORD *)(v21 + 40) = v8;
      v20[7] = v8;
      v8 = v20;
LABEL_31:
      if ( !v22 )
        goto LABEL_38;
      goto LABEL_32;
    }
    *(_QWORD *)(v5 + 48) = v4[1];
  }
  result = (_QWORD *)v4[1];
  if ( result )
    result[5] = a2[5];
  return result;
}
