// Function: sub_133EB60
// Address: 0x133eb60
//
_QWORD *__fastcall sub_133EB60(_QWORD *a1, _QWORD *a2)
{
  _QWORD *v3; // r13
  _QWORD *v4; // rbx
  __int64 v5; // r12
  _QWORD *v6; // rax
  __int64 v7; // rcx
  _QWORD *v8; // r8
  _QWORD *v9; // rdx
  _QWORD *v10; // rsi
  _QWORD *v11; // rbx
  unsigned __int64 v12; // rdi
  int v13; // ecx
  __int64 v14; // rcx
  _QWORD *v15; // r10
  __int64 v16; // rdx
  _QWORD *v17; // rax
  _QWORD *v18; // rcx
  _QWORD *v19; // r9
  _QWORD *v20; // rdi
  unsigned __int64 v21; // r14
  int v22; // edx
  __int64 v23; // rdx
  _QWORD *v24; // rdx
  __int64 v25; // rdi
  _QWORD *v26; // rsi
  unsigned __int64 v27; // r10
  int v28; // edi
  __int64 v29; // rdi
  _QWORD *result; // rax
  _QWORD *v32; // rcx
  _QWORD *v33; // rdx
  _QWORD *v34; // rax
  _QWORD *v35; // rdi
  _QWORD *v36; // r14
  unsigned __int64 v37; // r8
  int v38; // esi
  __int64 v39; // rsi
  _QWORD *v40; // r12
  __int64 v41; // rdx
  _QWORD *v42; // rax
  _QWORD *v43; // rsi
  _QWORD *v44; // r10
  _QWORD *v45; // r8
  unsigned __int64 v46; // r15
  int v47; // edx
  __int64 v48; // rdx
  __int64 v49; // rcx
  __int64 v50; // rax
  _QWORD *v51; // rax
  unsigned __int64 v52; // rsi
  int v53; // edx
  __int64 v54; // rdx
  _QWORD *v55; // r8
  _QWORD *v56; // rdx
  _QWORD *v57; // rdi
  _QWORD *v58; // r12
  unsigned __int64 v59; // rsi
  int v60; // ecx
  __int64 v61; // rcx
  _QWORD *v62; // rbx
  __int64 v63; // rdx
  _QWORD *v64; // rcx
  _QWORD *v65; // r10
  _QWORD *v66; // rsi
  unsigned __int64 v67; // r11
  int v68; // edx
  __int64 v69; // rdx
  _QWORD *v70; // rdx
  __int64 v71; // r8
  _QWORD *v72; // rdi
  unsigned __int64 v73; // r12
  int v74; // r8d
  __int64 v75; // r8
  __int64 v76; // rdx
  __int64 v77; // rsi
  _QWORD *v78; // rdx
  __int64 v79; // rdi
  _QWORD *v80; // rsi
  unsigned __int64 v81; // r11
  int v82; // edi
  __int64 v83; // rdi
  __int64 v84; // rcx

  v3 = a2 + 5;
  v4 = (_QWORD *)*a1;
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
    v32 = (_QWORD *)a2[6];
    if ( v32 )
    {
      a2[5] = 0;
      v33 = v32 + 5;
      *(_QWORD *)(*a1 + 48LL) = 0;
      v32[5] = 0;
      v34 = (_QWORD *)v32[6];
      if ( v34 )
      {
        v35 = (_QWORD *)v34[6];
        v36 = v34 + 5;
        if ( v35 )
          v35[5] = 0;
        *v33 = 0;
        v32[6] = 0;
        v34[5] = 0;
        v34[6] = 0;
        v37 = v32[2] & 0xFFFLL;
        v38 = (v37 > (v34[2] & 0xFFFuLL)) - (v37 < (v34[2] & 0xFFFuLL));
        if ( v37 > (v34[2] & 0xFFFuLL) == v37 < (v34[2] & 0xFFFuLL) )
          v38 = (v32 > v34) - (v32 < v34);
        if ( v38 == -1 )
        {
          v34[5] = v32;
          v77 = v32[7];
          v34[6] = v77;
          if ( v77 )
            *(_QWORD *)(v77 + 40) = v34;
          v32[7] = v34;
          v36 = v32 + 5;
        }
        else
        {
          *v33 = v34;
          v39 = v34[7];
          v32[6] = v39;
          if ( v39 )
            *(_QWORD *)(v39 + 40) = v32;
          v34[7] = v32;
          v32 = v34;
        }
        if ( v35 )
        {
          v40 = v32;
          while ( 1 )
          {
            v42 = (_QWORD *)v35[6];
            v43 = v35 + 5;
            if ( !v42 )
              break;
            v44 = (_QWORD *)v42[6];
            v45 = v42 + 5;
            if ( v44 )
              v44[5] = 0;
            *v43 = 0;
            v35[6] = 0;
            *v45 = 0;
            v42[6] = 0;
            v46 = v35[2] & 0xFFFLL;
            v47 = (v46 > (v42[2] & 0xFFFuLL)) - (v46 < (v42[2] & 0xFFFuLL));
            if ( v46 > (v42[2] & 0xFFFuLL) == v46 < (v42[2] & 0xFFFuLL) )
              v47 = (v35 > v42) - (v35 < v42);
            if ( v47 == -1 )
            {
              *v45 = v35;
              v48 = v35[7];
              v42[6] = v48;
              if ( v48 )
                *(_QWORD *)(v48 + 40) = v42;
              v35[7] = v42;
              v42 = v35;
            }
            else
            {
              *v43 = v42;
              v41 = v42[7];
              v35[6] = v41;
              if ( v41 )
                *(_QWORD *)(v41 + 40) = v35;
              v42[7] = v35;
            }
            v40[6] = v42;
            if ( !v44 )
              goto LABEL_129;
            v40 = v42;
            v35 = v44;
          }
          v40[6] = v35;
          v42 = v35;
        }
        else
        {
          v42 = v32;
        }
LABEL_129:
        v70 = (_QWORD *)v36[1];
        if ( v70 )
        {
          while ( 1 )
          {
            v72 = (_QWORD *)v70[6];
            v32[6] = 0;
            v70[6] = 0;
            if ( v70 )
            {
              v73 = v32[2] & 0xFFFLL;
              v74 = (v73 > (v70[2] & 0xFFFuLL)) - (v73 < (v70[2] & 0xFFFuLL));
              if ( v73 > (v70[2] & 0xFFFuLL) == v73 < (v70[2] & 0xFFFuLL) )
                v74 = (v70 < v32) - (v70 > v32);
              if ( v74 == -1 )
              {
                v70[5] = v32;
                v75 = v32[7];
                v70[6] = v75;
                if ( v75 )
                  *(_QWORD *)(v75 + 40) = v70;
                v32[7] = v70;
              }
              else
              {
                v32[5] = v70;
                v71 = v70[7];
                v32[6] = v71;
                if ( v71 )
                  *(_QWORD *)(v71 + 40) = v32;
                v70[7] = v32;
                v32 = v70;
              }
            }
            if ( !v72 )
              break;
            v42[6] = v32;
            v70 = (_QWORD *)v72[6];
            v42 = v32;
            v32 = v72;
          }
        }
      }
      v51 = (_QWORD *)*a1;
      if ( *a1 )
      {
        v52 = v51[2] & 0xFFFLL;
        v53 = (v52 > (v32[2] & 0xFFFuLL)) - (v52 < (v32[2] & 0xFFFuLL));
        if ( v52 > (v32[2] & 0xFFFuLL) == v52 < (v32[2] & 0xFFFuLL) )
          v53 = (v32 < v51) - (v32 > v51);
        if ( v53 == -1 )
        {
          v32[5] = v51;
          v76 = v51[7];
          v32[6] = v76;
          if ( v76 )
            *(_QWORD *)(v76 + 40) = v32;
          v51[7] = v32;
          v32 = v51;
        }
        else
        {
          v51[5] = v32;
          v54 = v32[7];
          v51[6] = v54;
          if ( v54 )
            *(_QWORD *)(v54 + 40) = v51;
          v32[7] = v51;
        }
      }
      *a1 = v32;
      if ( v4 != v32 )
        goto LABEL_2;
      result = v4 + 5;
    }
    v55 = (_QWORD *)result[2];
    if ( v55 )
    {
      result = (_QWORD *)v55[6];
      v56 = v55 + 5;
      if ( result )
      {
        v57 = (_QWORD *)result[6];
        v58 = result + 5;
        if ( v57 )
          v57[5] = 0;
        *v56 = 0;
        v55[6] = 0;
        result[5] = 0;
        result[6] = 0;
        v59 = v55[2] & 0xFFFLL;
        v60 = (v59 > (result[2] & 0xFFFuLL)) - (v59 < (result[2] & 0xFFFuLL));
        if ( v59 > (result[2] & 0xFFFuLL) == v59 < (result[2] & 0xFFFuLL) )
          v60 = (v55 > result) - (v55 < result);
        if ( v60 == -1 )
        {
          result[5] = v55;
          v84 = v55[7];
          result[6] = v84;
          if ( v84 )
            *(_QWORD *)(v84 + 40) = result;
          v55[7] = result;
          v58 = v55 + 5;
        }
        else
        {
          *v56 = result;
          v61 = result[7];
          v55[6] = v61;
          if ( v61 )
            *(_QWORD *)(v61 + 40) = v55;
          result[7] = v55;
          v55 = result;
        }
        if ( v57 )
        {
          v62 = v55;
          while ( 1 )
          {
            result = (_QWORD *)v57[6];
            v64 = v57 + 5;
            if ( !result )
              break;
            v65 = (_QWORD *)result[6];
            v66 = result + 5;
            if ( v65 )
              v65[5] = 0;
            *v64 = 0;
            v57[6] = 0;
            *v66 = 0;
            result[6] = 0;
            v67 = v57[2] & 0xFFFLL;
            v68 = (v67 > (result[2] & 0xFFFuLL)) - (v67 < (result[2] & 0xFFFuLL));
            if ( v67 > (result[2] & 0xFFFuLL) == v67 < (result[2] & 0xFFFuLL) )
              v68 = (v57 > result) - (v57 < result);
            if ( v68 == -1 )
            {
              *v66 = v57;
              v69 = v57[7];
              result[6] = v69;
              if ( v69 )
                *(_QWORD *)(v69 + 40) = result;
              v57[7] = result;
              result = v57;
            }
            else
            {
              *v64 = result;
              v63 = result[7];
              v57[6] = v63;
              if ( v63 )
                *(_QWORD *)(v63 + 40) = v57;
              result[7] = v57;
            }
            v62[6] = result;
            if ( !v65 )
              goto LABEL_156;
            v62 = result;
            v57 = v65;
          }
          v62[6] = v57;
          result = v57;
        }
        else
        {
          result = v55;
        }
LABEL_156:
        v78 = (_QWORD *)v58[1];
        if ( v78 )
        {
          while ( 1 )
          {
            v80 = (_QWORD *)v78[6];
            v55[6] = 0;
            v78[6] = 0;
            if ( v78 )
            {
              v81 = v55[2] & 0xFFFLL;
              v82 = (v81 > (v78[2] & 0xFFFuLL)) - (v81 < (v78[2] & 0xFFFuLL));
              if ( v81 > (v78[2] & 0xFFFuLL) == v81 < (v78[2] & 0xFFFuLL) )
                v82 = (v78 < v55) - (v78 > v55);
              if ( v82 == -1 )
              {
                v78[5] = v55;
                v83 = v55[7];
                v78[6] = v83;
                if ( v83 )
                  *(_QWORD *)(v83 + 40) = v78;
                v55[7] = v78;
              }
              else
              {
                v55[5] = v78;
                v79 = v78[7];
                v55[6] = v79;
                if ( v79 )
                  *(_QWORD *)(v79 + 40) = v55;
                v78[7] = v55;
                v55 = v78;
              }
            }
            if ( !v80 )
              break;
            result[6] = v55;
            v78 = (_QWORD *)v80[6];
            result = v55;
            v55 = v80;
          }
        }
      }
    }
    *a1 = v55;
    return result;
  }
LABEL_2:
  v5 = a2[5];
  v6 = (_QWORD *)v3[2];
  if ( v5 && (v7 = v5 + 40, a2 == *(_QWORD **)(v5 + 56)) )
  {
    if ( v6 )
    {
      v8 = (_QWORD *)v6[6];
      v9 = v6 + 5;
      if ( !v8 )
      {
        v11 = v6 + 5;
        v8 = (_QWORD *)v3[2];
        goto LABEL_47;
      }
      goto LABEL_7;
    }
    v50 = v3[1];
    *(_QWORD *)(v5 + 56) = v50;
    if ( v50 )
      *(_QWORD *)(v50 + 40) = v5;
  }
  else
  {
    if ( v6 )
    {
      v8 = (_QWORD *)v6[6];
      v9 = v6 + 5;
      if ( !v8 )
        goto LABEL_86;
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
      v12 = v6[2] & 0xFFFLL;
      v13 = (v12 > (v8[2] & 0xFFFuLL)) - (v12 < (v8[2] & 0xFFFuLL));
      if ( v12 > (v8[2] & 0xFFFuLL) == v12 < (v8[2] & 0xFFFuLL) )
        v13 = (v8 < v6) - (v8 > v6);
      if ( v13 == -1 )
      {
        *v11 = v6;
        v49 = v9[2];
        v8[6] = v49;
        if ( v49 )
          *(_QWORD *)(v49 + 40) = v8;
        v9[2] = v8;
        v11 = v9;
        v8 = v6;
      }
      else
      {
        *v9 = v8;
        v14 = v8[7];
        v9[1] = v14;
        if ( v14 )
          *(_QWORD *)(v14 + 40) = v6;
        v8[7] = v6;
      }
      if ( v10 )
      {
        v15 = v8;
        while ( 1 )
        {
          v17 = (_QWORD *)v10[6];
          v18 = v10 + 5;
          if ( !v17 )
            break;
          v19 = (_QWORD *)v17[6];
          v20 = v17 + 5;
          if ( v19 )
            v19[5] = 0;
          *v18 = 0;
          v10[6] = 0;
          *v20 = 0;
          v17[6] = 0;
          v21 = v10[2] & 0xFFFLL;
          v22 = (v21 > (v17[2] & 0xFFFuLL)) - (v21 < (v17[2] & 0xFFFuLL));
          if ( v21 > (v17[2] & 0xFFFuLL) == v21 < (v17[2] & 0xFFFuLL) )
            v22 = (v10 > v17) - (v10 < v17);
          if ( v22 == -1 )
          {
            *v20 = v10;
            v23 = v10[7];
            v17[6] = v23;
            if ( v23 )
              *(_QWORD *)(v23 + 40) = v17;
            v10[7] = v17;
            v17 = v10;
            v15[6] = v10;
            if ( !v19 )
              goto LABEL_30;
          }
          else
          {
            *v18 = v17;
            v16 = v17[7];
            v10[6] = v16;
            if ( v16 )
              *(_QWORD *)(v16 + 40) = v10;
            v17[7] = v10;
            v15[6] = v17;
            if ( !v19 )
              goto LABEL_30;
          }
          v15 = v17;
          v10 = v19;
        }
        v15[6] = v10;
        v17 = v10;
      }
      else
      {
        v17 = v8;
      }
LABEL_30:
      v24 = (_QWORD *)v11[1];
      if ( !v24 )
      {
LABEL_45:
        if ( v5 )
        {
          v7 = v5 + 40;
LABEL_47:
          *v11 = v5;
          *(_QWORD *)(v7 + 16) = v8;
LABEL_48:
          result = (_QWORD *)v3[1];
          v11[1] = result;
          if ( result )
            result[5] = v8;
          return result;
        }
        v5 = a2[5];
        v9 = v11;
        v6 = v8;
LABEL_86:
        *v9 = v5;
        if ( v5 )
          *(_QWORD *)(v5 + 48) = v6;
        v11 = v9;
        v8 = v6;
        goto LABEL_48;
      }
      while ( 1 )
      {
        v26 = (_QWORD *)v24[6];
        v8[6] = 0;
        v24[6] = 0;
        if ( !v24 )
          goto LABEL_35;
        v27 = v8[2] & 0xFFFLL;
        v28 = (v27 > (v24[2] & 0xFFFuLL)) - (v27 < (v24[2] & 0xFFFuLL));
        if ( v27 > (v24[2] & 0xFFFuLL) == v27 < (v24[2] & 0xFFFuLL) )
          v28 = (v24 < v8) - (v24 > v8);
        if ( v28 != -1 )
          break;
        v24[5] = v8;
        v29 = v8[7];
        v24[6] = v29;
        if ( v29 )
          *(_QWORD *)(v29 + 40) = v24;
        v8[7] = v24;
        if ( !v26 )
        {
LABEL_44:
          v11 = v8 + 5;
          goto LABEL_45;
        }
LABEL_36:
        v17[6] = v8;
        v24 = (_QWORD *)v26[6];
        v17 = v8;
        v8 = v26;
      }
      v8[5] = v24;
      v25 = v24[7];
      v8[6] = v25;
      if ( v25 )
        *(_QWORD *)(v25 + 40) = v8;
      v24[7] = v8;
      v8 = v24;
LABEL_35:
      if ( !v26 )
        goto LABEL_44;
      goto LABEL_36;
    }
    *(_QWORD *)(v5 + 48) = v3[1];
  }
  result = (_QWORD *)v3[1];
  if ( result )
    result[5] = a2[5];
  return result;
}
