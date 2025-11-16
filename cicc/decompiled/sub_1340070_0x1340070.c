// Function: sub_1340070
// Address: 0x1340070
//
_QWORD *__fastcall sub_1340070(_QWORD *a1, _QWORD *a2)
{
  _QWORD *v2; // r12
  __int64 v3; // r15
  _QWORD *v4; // rax
  __int64 v5; // rcx
  _QWORD *v6; // r9
  _QWORD *v7; // rdx
  _QWORD *v8; // rcx
  _QWORD *v9; // r14
  int v10; // edi
  __int64 v11; // rdi
  _QWORD *v12; // r11
  __int64 v13; // rdx
  _QWORD *v14; // rax
  _QWORD *v15; // rdi
  _QWORD *v16; // r10
  _QWORD *v17; // r8
  int v18; // edx
  __int64 v19; // rdx
  _QWORD *v20; // rdx
  __int64 v21; // r8
  _QWORD *v22; // rdi
  unsigned __int64 v23; // r14
  int v24; // r8d
  __int64 v25; // r8
  _QWORD *result; // rax
  _QWORD *v28; // rcx
  _QWORD *v29; // rdx
  _QWORD *v30; // rax
  _QWORD *v31; // rdi
  unsigned __int64 v32; // r15
  int v33; // r8d
  __int64 v34; // r8
  _QWORD *v35; // r13
  __int64 v36; // rdx
  _QWORD *v37; // rax
  _QWORD *v38; // r8
  _QWORD *v39; // r11
  _QWORD *v40; // r9
  int v41; // edx
  __int64 v42; // rdx
  __int64 v43; // rdi
  __int64 v44; // rax
  _QWORD *v45; // rax
  int v46; // edx
  __int64 v47; // rdx
  _QWORD *v48; // r8
  _QWORD *v49; // rdx
  _QWORD *v50; // rcx
  _QWORD *v51; // r12
  unsigned __int64 v52; // rbx
  int v53; // esi
  __int64 v54; // rsi
  _QWORD *v55; // r11
  __int64 v56; // rdx
  _QWORD *v57; // rsi
  _QWORD *v58; // r9
  _QWORD *v59; // rdi
  int v60; // edx
  __int64 v61; // rdx
  _QWORD *v62; // rdx
  __int64 v63; // r9
  _QWORD *v64; // r8
  unsigned __int64 v65; // r15
  int v66; // r9d
  __int64 v67; // r9
  __int64 v68; // rdx
  __int64 v69; // r8
  _QWORD *v70; // rdx
  __int64 v71; // rdi
  _QWORD *v72; // rsi
  int v73; // edi
  __int64 v74; // rdi
  __int64 v75; // rsi
  _QWORD *v76; // [rsp+0h] [rbp-30h]

  v2 = (_QWORD *)*a1;
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
    v28 = (_QWORD *)a2[6];
    if ( v28 )
    {
      a2[5] = 0;
      v29 = v28 + 5;
      *(_QWORD *)(*a1 + 48LL) = 0;
      v28[5] = 0;
      v30 = (_QWORD *)v28[6];
      if ( v30 )
      {
        v76 = v30 + 5;
        v31 = (_QWORD *)v30[6];
        if ( v31 )
          v31[5] = 0;
        *v29 = 0;
        v28[6] = 0;
        v30[5] = 0;
        v30[6] = 0;
        v32 = v30[4];
        v33 = (v28[4] > v32) - (v28[4] < v32);
        if ( v28[4] > v32 == v28[4] < v32 )
          v33 = (v28[1] > v30[1]) - (v28[1] < v30[1]);
        if ( v33 == -1 )
        {
          v30[5] = v28;
          v69 = v28[7];
          v30[6] = v69;
          if ( v69 )
            *(_QWORD *)(v69 + 40) = v30;
          v28[7] = v30;
          v76 = v28 + 5;
        }
        else
        {
          *v29 = v30;
          v34 = v30[7];
          v28[6] = v34;
          if ( v34 )
            *(_QWORD *)(v34 + 40) = v28;
          v30[7] = v28;
          v28 = v30;
        }
        if ( v31 )
        {
          v35 = v28;
          while ( 1 )
          {
            v37 = (_QWORD *)v31[6];
            v38 = v31 + 5;
            if ( !v37 )
              break;
            v39 = (_QWORD *)v37[6];
            v40 = v37 + 5;
            if ( v39 )
              v39[5] = 0;
            *v38 = 0;
            v31[6] = 0;
            *v40 = 0;
            v37[6] = 0;
            v41 = (v31[4] > v37[4]) - (v31[4] < v37[4]);
            if ( v31[4] > v37[4] == v31[4] < v37[4] )
              v41 = (v31[1] > v37[1]) - (v31[1] < v37[1]);
            if ( v41 == -1 )
            {
              *v40 = v31;
              v42 = v31[7];
              v37[6] = v42;
              if ( v42 )
                *(_QWORD *)(v42 + 40) = v37;
              v31[7] = v37;
              v37 = v31;
            }
            else
            {
              *v38 = v37;
              v36 = v37[7];
              v31[6] = v36;
              if ( v36 )
                *(_QWORD *)(v36 + 40) = v31;
              v37[7] = v31;
            }
            v35[6] = v37;
            if ( !v39 )
              goto LABEL_129;
            v35 = v37;
            v31 = v39;
          }
          v35[6] = v31;
          v37 = v31;
        }
        else
        {
          v37 = v28;
        }
LABEL_129:
        v62 = (_QWORD *)v76[1];
        if ( v62 )
        {
          while ( 1 )
          {
            v64 = (_QWORD *)v62[6];
            v28[6] = 0;
            v62[6] = 0;
            if ( v62 )
            {
              v65 = v62[4];
              v66 = (v28[4] > v65) - (v28[4] < v65);
              if ( v28[4] > v65 == v28[4] < v65 )
                v66 = (v28[1] > v62[1]) - (v28[1] < v62[1]);
              if ( v66 == -1 )
              {
                v62[5] = v28;
                v67 = v28[7];
                v62[6] = v67;
                if ( v67 )
                  *(_QWORD *)(v67 + 40) = v62;
                v28[7] = v62;
              }
              else
              {
                v28[5] = v62;
                v63 = v62[7];
                v28[6] = v63;
                if ( v63 )
                  *(_QWORD *)(v63 + 40) = v28;
                v62[7] = v28;
                v28 = v62;
              }
            }
            if ( !v64 )
              break;
            v37[6] = v28;
            v62 = (_QWORD *)v64[6];
            v37 = v28;
            v28 = v64;
          }
        }
      }
      v45 = (_QWORD *)*a1;
      if ( *a1 )
      {
        v46 = (v45[4] > v28[4]) - (v45[4] < v28[4]);
        if ( v45[4] > v28[4] == v45[4] < v28[4] )
          v46 = (v45[1] > v28[1]) - (v45[1] < v28[1]);
        if ( v46 == -1 )
        {
          v28[5] = v45;
          v68 = v45[7];
          v28[6] = v68;
          if ( v68 )
            *(_QWORD *)(v68 + 40) = v28;
          v45[7] = v28;
          v28 = v45;
        }
        else
        {
          v45[5] = v28;
          v47 = v28[7];
          v45[6] = v47;
          if ( v47 )
            *(_QWORD *)(v47 + 40) = v45;
          v28[7] = v45;
        }
      }
      *a1 = v28;
      if ( v2 != v28 )
        goto LABEL_2;
      result = v2 + 5;
    }
    v48 = (_QWORD *)result[2];
    if ( v48 )
    {
      result = (_QWORD *)v48[6];
      v49 = v48 + 5;
      if ( result )
      {
        v50 = (_QWORD *)result[6];
        v51 = result + 5;
        if ( v50 )
          v50[5] = 0;
        *v49 = 0;
        v48[6] = 0;
        result[5] = 0;
        result[6] = 0;
        v52 = result[4];
        v53 = (v48[4] > v52) - (v48[4] < v52);
        if ( v48[4] > v52 == v48[4] < v52 )
          v53 = (v48[1] > result[1]) - (v48[1] < result[1]);
        if ( v53 == -1 )
        {
          result[5] = v48;
          v75 = v48[7];
          result[6] = v75;
          if ( v75 )
            *(_QWORD *)(v75 + 40) = result;
          v48[7] = result;
          v51 = v48 + 5;
        }
        else
        {
          *v49 = result;
          v54 = result[7];
          v48[6] = v54;
          if ( v54 )
            *(_QWORD *)(v54 + 40) = v48;
          result[7] = v48;
          v48 = result;
        }
        if ( v50 )
        {
          v55 = v48;
          while ( 1 )
          {
            result = (_QWORD *)v50[6];
            v57 = v50 + 5;
            if ( !result )
              break;
            v58 = (_QWORD *)result[6];
            v59 = result + 5;
            if ( v58 )
              v58[5] = 0;
            *v57 = 0;
            v50[6] = 0;
            *v59 = 0;
            result[6] = 0;
            v60 = (v50[4] > result[4]) - (v50[4] < result[4]);
            if ( v50[4] > result[4] == v50[4] < result[4] )
              v60 = (v50[1] > result[1]) - (v50[1] < result[1]);
            if ( v60 == -1 )
            {
              *v59 = v50;
              v61 = v50[7];
              result[6] = v61;
              if ( v61 )
                *(_QWORD *)(v61 + 40) = result;
              v50[7] = result;
              result = v50;
            }
            else
            {
              *v57 = result;
              v56 = result[7];
              v50[6] = v56;
              if ( v56 )
                *(_QWORD *)(v56 + 40) = v50;
              result[7] = v50;
            }
            v55[6] = result;
            if ( !v58 )
              goto LABEL_156;
            v55 = result;
            v50 = v58;
          }
          v55[6] = v50;
          result = v50;
        }
        else
        {
          result = v48;
        }
LABEL_156:
        v70 = (_QWORD *)v51[1];
        if ( v70 )
        {
          while ( 1 )
          {
            v72 = (_QWORD *)v70[6];
            v48[6] = 0;
            v70[6] = 0;
            if ( v70 )
            {
              v73 = (v48[4] > v70[4]) - (v48[4] < v70[4]);
              if ( v48[4] > v70[4] == v48[4] < v70[4] )
                v73 = (v48[1] > v70[1]) - (v48[1] < v70[1]);
              if ( v73 == -1 )
              {
                v70[5] = v48;
                v74 = v48[7];
                v70[6] = v74;
                if ( v74 )
                  *(_QWORD *)(v74 + 40) = v70;
                v48[7] = v70;
              }
              else
              {
                v48[5] = v70;
                v71 = v70[7];
                v48[6] = v71;
                if ( v71 )
                  *(_QWORD *)(v71 + 40) = v48;
                v70[7] = v48;
                v48 = v70;
              }
            }
            if ( !v72 )
              break;
            result[6] = v48;
            v70 = (_QWORD *)v72[6];
            result = v48;
            v48 = v72;
          }
        }
      }
    }
    *a1 = v48;
    return result;
  }
LABEL_2:
  v3 = a2[5];
  v4 = (_QWORD *)a2[7];
  if ( v3 && (v5 = v3 + 40, a2 == *(_QWORD **)(v3 + 56)) )
  {
    if ( v4 )
    {
      v6 = (_QWORD *)v4[6];
      v7 = v4 + 5;
      if ( !v6 )
      {
        v9 = v4 + 5;
        v6 = (_QWORD *)a2[7];
        goto LABEL_47;
      }
      goto LABEL_7;
    }
    v44 = a2[6];
    *(_QWORD *)(v3 + 56) = v44;
    if ( v44 )
      *(_QWORD *)(v44 + 40) = v3;
  }
  else
  {
    if ( v4 )
    {
      v6 = (_QWORD *)v4[6];
      v7 = v4 + 5;
      if ( !v6 )
        goto LABEL_86;
      v3 = 0;
LABEL_7:
      v8 = (_QWORD *)v6[6];
      v9 = v6 + 5;
      if ( v8 )
        v8[5] = 0;
      *v7 = 0;
      v7[1] = 0;
      *v9 = 0;
      v6[6] = 0;
      v10 = (v4[4] > v6[4]) - (v4[4] < v6[4]);
      if ( v4[4] > v6[4] == v4[4] < v6[4] )
        v10 = (v4[1] > v6[1]) - (v4[1] < v6[1]);
      if ( v10 == -1 )
      {
        *v9 = v4;
        v43 = v7[2];
        v6[6] = v43;
        if ( v43 )
          *(_QWORD *)(v43 + 40) = v6;
        v7[2] = v6;
        v9 = v7;
        v6 = v4;
      }
      else
      {
        *v7 = v6;
        v11 = v6[7];
        v7[1] = v11;
        if ( v11 )
          *(_QWORD *)(v11 + 40) = v4;
        v6[7] = v4;
      }
      if ( v8 )
      {
        v12 = v6;
        while ( 1 )
        {
          v14 = (_QWORD *)v8[6];
          v15 = v8 + 5;
          if ( !v14 )
            break;
          v16 = (_QWORD *)v14[6];
          v17 = v14 + 5;
          if ( v16 )
            v16[5] = 0;
          *v15 = 0;
          v8[6] = 0;
          *v17 = 0;
          v14[6] = 0;
          v18 = (v8[4] > v14[4]) - (v8[4] < v14[4]);
          if ( v8[4] > v14[4] == v8[4] < v14[4] )
            v18 = (v8[1] > v14[1]) - (v8[1] < v14[1]);
          if ( v18 == -1 )
          {
            *v17 = v8;
            v19 = v8[7];
            v14[6] = v19;
            if ( v19 )
              *(_QWORD *)(v19 + 40) = v14;
            v8[7] = v14;
            v14 = v8;
            v12[6] = v8;
            if ( !v16 )
              goto LABEL_30;
          }
          else
          {
            *v15 = v14;
            v13 = v14[7];
            v8[6] = v13;
            if ( v13 )
              *(_QWORD *)(v13 + 40) = v8;
            v14[7] = v8;
            v12[6] = v14;
            if ( !v16 )
              goto LABEL_30;
          }
          v12 = v14;
          v8 = v16;
        }
        v12[6] = v8;
        v14 = v8;
      }
      else
      {
        v14 = v6;
      }
LABEL_30:
      v20 = (_QWORD *)v9[1];
      if ( !v20 )
      {
LABEL_45:
        if ( v3 )
        {
          v5 = v3 + 40;
LABEL_47:
          *v9 = v3;
          *(_QWORD *)(v5 + 16) = v6;
LABEL_48:
          result = (_QWORD *)a2[6];
          v9[1] = result;
          if ( result )
            result[5] = v6;
          return result;
        }
        v3 = a2[5];
        v7 = v9;
        v4 = v6;
LABEL_86:
        *v7 = v3;
        if ( v3 )
          *(_QWORD *)(v3 + 48) = v4;
        v9 = v7;
        v6 = v4;
        goto LABEL_48;
      }
      while ( 1 )
      {
        v22 = (_QWORD *)v20[6];
        v6[6] = 0;
        v20[6] = 0;
        if ( !v20 )
          goto LABEL_35;
        v23 = v20[4];
        v24 = (v6[4] > v23) - (v6[4] < v23);
        if ( v6[4] > v23 == v6[4] < v23 )
          v24 = (v6[1] > v20[1]) - (v6[1] < v20[1]);
        if ( v24 != -1 )
          break;
        v20[5] = v6;
        v25 = v6[7];
        v20[6] = v25;
        if ( v25 )
          *(_QWORD *)(v25 + 40) = v20;
        v6[7] = v20;
        if ( !v22 )
        {
LABEL_44:
          v9 = v6 + 5;
          goto LABEL_45;
        }
LABEL_36:
        v14[6] = v6;
        v20 = (_QWORD *)v22[6];
        v14 = v6;
        v6 = v22;
      }
      v6[5] = v20;
      v21 = v20[7];
      v6[6] = v21;
      if ( v21 )
        *(_QWORD *)(v21 + 40) = v6;
      v20[7] = v6;
      v6 = v20;
LABEL_35:
      if ( !v22 )
        goto LABEL_44;
      goto LABEL_36;
    }
    *(_QWORD *)(v3 + 48) = a2[6];
  }
  result = (_QWORD *)a2[6];
  if ( result )
    result[5] = a2[5];
  return result;
}
