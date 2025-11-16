// Function: sub_100B250
// Address: 0x100b250
//
__int64 *__fastcall sub_100B250(__int64 *a1, _BYTE *a2)
{
  char v3; // al
  __int64 v4; // r14
  char v5; // dl
  bool v6; // zf
  char v8; // al
  __int64 *v9; // rsi
  char v10; // al
  __int64 **v11; // rsi
  __int64 **v12; // rax
  __int64 *v13; // rdx
  __int64 *v14; // rdx
  __int64 v15; // rsi
  __int64 v16; // rdx
  _BYTE *v17; // rsi
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rsi
  _BYTE *v22; // rax
  __int64 v23; // rdx
  _BYTE *v24; // r15
  __int64 *v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rcx
  _BYTE *v28; // rdx
  _BYTE *v29; // rdx
  _BYTE *v30; // rdx
  unsigned __int8 *v31; // rdx
  __int64 v32; // rax
  __int64 v33; // rdx
  __int64 v34; // rcx
  _BYTE *v35; // r14
  unsigned __int8 *v36; // rdx
  __int64 *v37; // rax
  char v38; // al
  _BYTE *v39; // rsi
  _BYTE *v40; // rax
  char v41; // al
  char v42; // al
  _BYTE *v43; // rsi
  _BYTE *v44; // rax
  __int64 v45; // rdx
  __int64 v46; // rax
  __int64 v47; // rax
  __int64 v48; // rsi
  __int64 v49; // rax
  __int64 v50; // rcx
  char v51; // al
  __int64 v52; // rsi
  char v53; // al
  _BYTE *v54; // rsi
  _BYTE *v55; // rsi
  char v56; // al
  __int64 v57; // rsi
  __int64 v58; // rax
  __int64 v59; // rdx
  __int64 v60; // rcx
  char v61; // al
  __int64 v62; // rsi
  char v63; // al
  __int64 v64; // rsi
  char v65; // al
  __int64 v66; // rsi
  __int64 v67; // rax
  __int64 v68; // rax
  __int64 v69; // rax
  __int64 v70; // rax
  __int64 v71; // rax
  __int64 v72; // rax
  __int64 v73; // rax
  __int64 v74; // rax
  __int64 v75; // rsi
  __int64 v76; // rax
  __int64 v77; // rdx
  _BYTE *v78; // [rsp+0h] [rbp-A0h]
  __int64 v79; // [rsp+0h] [rbp-A0h]
  __int64 v80; // [rsp+0h] [rbp-A0h]
  __int64 v81; // [rsp+18h] [rbp-88h] BYREF
  __int64 v82; // [rsp+20h] [rbp-80h] BYREF
  __int64 v83; // [rsp+28h] [rbp-78h] BYREF
  _QWORD *v84; // [rsp+30h] [rbp-70h] BYREF
  __int64 v85; // [rsp+38h] [rbp-68h] BYREF
  __int64 v86; // [rsp+40h] [rbp-60h]
  __int64 *v87; // [rsp+50h] [rbp-50h] BYREF
  __int64 *v88; // [rsp+58h] [rbp-48h] BYREF
  __int64 *v89; // [rsp+60h] [rbp-40h]
  __int64 **v90; // [rsp+68h] [rbp-38h]

  v3 = *a2;
  v4 = a1[1];
  v88 = a1;
  v87 = 0;
  if ( v3 == 59 )
  {
    v8 = sub_995B10(&v87, *((_QWORD *)a2 - 8));
    v9 = (__int64 *)*((_QWORD *)a2 - 4);
    if ( (!v8 || v9 != v88) && (!(unsigned __int8)sub_995B10(&v87, (__int64)v9) || *((__int64 **)a2 - 8) != v88) )
    {
      v3 = *a2;
      v87 = 0;
      v88 = a1;
      if ( v3 != 59 )
        goto LABEL_2;
      v10 = sub_995B10(&v87, *((_QWORD *)a2 - 8));
      v11 = (__int64 **)*((_QWORD *)a2 - 4);
      if ( !v10 || *(_BYTE *)v11 != 57 || *(v11 - 8) != v88 && v88 != *(v11 - 4) )
      {
        if ( !(unsigned __int8)sub_995B10(&v87, (__int64)v11)
          || (v12 = (__int64 **)*((_QWORD *)a2 - 8), *(_BYTE *)v12 != 57)
          || *(v12 - 8) != v88 && v88 != *(v12 - 4) )
        {
          v3 = *a2;
          if ( *a2 != 57 )
            goto LABEL_3;
          goto LABEL_21;
        }
      }
    }
    return (__int64 *)sub_AD62B0(v4);
  }
LABEL_2:
  if ( v3 != 57 )
    goto LABEL_3;
LABEL_21:
  v13 = (__int64 *)*((_QWORD *)a2 - 8);
  if ( a1 == v13 && v13 )
    return a1;
  v14 = (__int64 *)*((_QWORD *)a2 - 4);
  if ( a1 == v14 )
  {
    if ( v14 )
      return a1;
  }
LABEL_3:
  v5 = *(_BYTE *)a1;
  if ( *(_BYTE *)a1 != 59 )
    goto LABEL_4;
  v15 = *(a1 - 8);
  if ( v15 )
  {
    v16 = *(a1 - 4);
    v81 = *(a1 - 8);
    if ( v16 )
    {
      v82 = v16;
      if ( v3 == 58 )
      {
        v49 = *((_QWORD *)a2 - 8);
        v50 = *((_QWORD *)a2 - 4);
        if ( v15 == v49 && v16 == v50 )
          return (__int64 *)a2;
        if ( v15 == v50 && v16 == v49 )
          return (__int64 *)a2;
      }
    }
  }
  v87 = 0;
  v88 = &v81;
  v89 = &v82;
  if ( !(unsigned __int8)sub_995B10(&v87, v15) )
  {
LABEL_44:
    v17 = (_BYTE *)*(a1 - 4);
    goto LABEL_45;
  }
  v17 = (_BYTE *)*(a1 - 4);
  if ( *v17 == 59 )
  {
    v18 = *((_QWORD *)v17 - 8);
    if ( v18 )
    {
      *v88 = v18;
      v19 = *((_QWORD *)v17 - 4);
      if ( v19 )
      {
LABEL_35:
        *v89 = v19;
        if ( *a2 == 58 )
        {
          v20 = *((_QWORD *)a2 - 8);
          v21 = *((_QWORD *)a2 - 4);
          if ( v81 == v20 && v82 == v21 )
            return (__int64 *)sub_AD62B0(v4);
          if ( v82 == v20 && v81 == v21 )
            return (__int64 *)sub_AD62B0(v4);
        }
        goto LABEL_49;
      }
      goto LABEL_44;
    }
  }
LABEL_45:
  if ( (unsigned __int8)sub_995B10(&v87, (__int64)v17) )
  {
    v22 = (_BYTE *)*(a1 - 8);
    if ( *v22 == 59 )
    {
      v23 = *((_QWORD *)v22 - 8);
      if ( v23 )
      {
        *v88 = v23;
        v19 = *((_QWORD *)v22 - 4);
        if ( v19 )
          goto LABEL_35;
      }
    }
  }
LABEL_49:
  v5 = *(_BYTE *)a1;
LABEL_4:
  v87 = &v81;
  v88 = 0;
  v89 = &v82;
  if ( v5 != 57 )
    goto LABEL_5;
  v24 = (_BYTE *)*(a1 - 4);
  if ( !*(a1 - 8) )
    goto LABEL_144;
  v81 = *(a1 - 8);
  v25 = &v81;
  if ( *v24 != 59 )
    goto LABEL_52;
  v63 = sub_995B10(&v88, *((_QWORD *)v24 - 8));
  v64 = *((_QWORD *)v24 - 4);
  if ( !v63 || !v64 )
  {
    if ( (unsigned __int8)sub_995B10(&v88, v64) )
    {
      v69 = *((_QWORD *)v24 - 8);
      if ( v69 )
      {
        *v89 = v69;
        goto LABEL_53;
      }
    }
    v24 = (_BYTE *)*(a1 - 4);
LABEL_144:
    if ( !v24 )
      goto LABEL_58;
    v25 = v87;
LABEL_52:
    *v25 = (__int64)v24;
    if ( !(unsigned __int8)sub_996420(&v88, 30, (unsigned __int8 *)*(a1 - 8)) )
      goto LABEL_58;
    goto LABEL_53;
  }
  *v89 = v64;
LABEL_53:
  if ( *a2 == 59 )
  {
    v26 = *((_QWORD *)a2 - 8);
    v27 = *((_QWORD *)a2 - 4);
    if ( v81 == v26 && v82 == v27 )
      return (__int64 *)a2;
    if ( v81 == v27 && v82 == v26 )
      return (__int64 *)a2;
  }
LABEL_58:
  v5 = *(_BYTE *)a1;
LABEL_5:
  v87 = 0;
  v88 = &v81;
  v89 = &v82;
  if ( v5 != 59 )
    goto LABEL_6;
  v28 = (_BYTE *)*(a1 - 8);
  if ( *v28 != 59 )
    goto LABEL_60;
  v79 = *(a1 - 8);
  v61 = sub_995B10(&v87, *((_QWORD *)v28 - 8));
  v62 = *(_QWORD *)(v79 - 32);
  if ( v61 && v62 )
  {
    *v88 = v62;
  }
  else
  {
    if ( !(unsigned __int8)sub_995B10(&v87, v62) || (v67 = *(_QWORD *)(v79 - 64)) == 0 )
    {
LABEL_60:
      v29 = (_BYTE *)*(a1 - 4);
      goto LABEL_61;
    }
    *v88 = v67;
  }
  v29 = (_BYTE *)*(a1 - 4);
  if ( v29 )
  {
    *v89 = (__int64)v29;
    goto LABEL_117;
  }
LABEL_61:
  if ( *v29 == 59 )
  {
    v78 = v29;
    v56 = sub_995B10(&v87, *((_QWORD *)v29 - 8));
    v57 = *((_QWORD *)v78 - 4);
    if ( v56 && v57 )
    {
      *v88 = v57;
    }
    else
    {
      if ( !(unsigned __int8)sub_995B10(&v87, v57) )
        goto LABEL_62;
      v71 = *((_QWORD *)v78 - 8);
      if ( !v71 )
        goto LABEL_62;
      *v88 = v71;
    }
    v58 = *(a1 - 8);
    if ( v58 )
    {
      *v89 = v58;
LABEL_117:
      if ( *a2 != 57 )
        goto LABEL_62;
      v59 = *((_QWORD *)a2 - 8);
      v60 = *((_QWORD *)a2 - 4);
      if ( (v81 != v59 || v82 != v60) && (v81 != v60 || v82 != v59) )
        goto LABEL_62;
      return a1;
    }
  }
LABEL_62:
  v5 = *(_BYTE *)a1;
LABEL_6:
  v87 = 0;
  v88 = &v81;
  v89 = &v82;
  if ( v5 != 58 )
    goto LABEL_7;
  v30 = (_BYTE *)*(a1 - 8);
  if ( *v30 == 59 )
  {
    v80 = *(a1 - 8);
    v65 = sub_995B10(&v87, *((_QWORD *)v30 - 8));
    v66 = *(_QWORD *)(v80 - 32);
    if ( v65 && v66 )
    {
      *v88 = v66;
    }
    else
    {
      if ( !(unsigned __int8)sub_995B10(&v87, v66) || (v68 = *(_QWORD *)(v80 - 64)) == 0 )
      {
        v31 = (unsigned __int8 *)*(a1 - 4);
        goto LABEL_65;
      }
      *v88 = v68;
    }
    v31 = (unsigned __int8 *)*(a1 - 4);
    if ( v31 )
    {
      *v89 = (__int64)v31;
      goto LABEL_68;
    }
  }
  else
  {
    v31 = (unsigned __int8 *)*(a1 - 4);
  }
LABEL_65:
  if ( (unsigned __int8)sub_996420(&v87, 30, v31) )
  {
    v32 = *(a1 - 8);
    if ( v32 )
    {
      *v89 = v32;
LABEL_68:
      if ( *a2 != 59 )
        goto LABEL_72;
      v33 = *((_QWORD *)a2 - 8);
      v34 = *((_QWORD *)a2 - 4);
      if ( (v81 != v33 || v82 != v34) && (v81 != v34 || v82 != v33) )
        goto LABEL_72;
      return (__int64 *)sub_AD62B0(v4);
    }
  }
LABEL_72:
  v5 = *(_BYTE *)a1;
LABEL_7:
  v88 = 0;
  v87 = &v83;
  v89 = &v81;
  v90 = (__int64 **)&v82;
  if ( v5 != 57 )
    goto LABEL_8;
  v35 = (_BYTE *)*(a1 - 8);
  if ( !v35 )
    goto LABEL_75;
  v83 = *(a1 - 8);
  if ( *v35 != 59 )
    goto LABEL_75;
  v51 = sub_995B10(&v88, *((_QWORD *)v35 - 8));
  v52 = *((_QWORD *)v35 - 4);
  if ( v51 && v52 )
  {
    *v89 = v52;
  }
  else
  {
    if ( !(unsigned __int8)sub_995B10(&v88, v52) || (v70 = *((_QWORD *)v35 - 8)) == 0 )
    {
LABEL_75:
      v36 = (unsigned __int8 *)*(a1 - 4);
      if ( !v36 )
        goto LABEL_8;
      *v87 = (__int64)v36;
      if ( !(unsigned __int8)sub_996420(&v88, 30, v36) )
        goto LABEL_8;
      v37 = (__int64 *)*(a1 - 8);
      if ( !v37 )
        goto LABEL_8;
LABEL_103:
      *v90 = v37;
      v6 = *a2 == 59;
      v84 = 0;
      v85 = v81;
      v86 = v82;
      if ( v6 )
      {
        v53 = sub_995B10(&v84, *((_QWORD *)a2 - 8));
        v54 = (_BYTE *)*((_QWORD *)a2 - 4);
        if ( v53 )
        {
          if ( *v54 == 58 && (unsigned __int8)sub_FFE640(&v85, (__int64)v54) )
            return (__int64 *)v83;
        }
        if ( (unsigned __int8)sub_995B10(&v84, (__int64)v54) )
        {
          v55 = (_BYTE *)*((_QWORD *)a2 - 8);
          if ( *v55 == 58 )
          {
            if ( (unsigned __int8)sub_FFE640(&v85, (__int64)v55) )
              return (__int64 *)v83;
          }
        }
      }
      goto LABEL_8;
    }
    *v89 = v70;
  }
  v37 = (__int64 *)*(a1 - 4);
  if ( v37 )
    goto LABEL_103;
LABEL_8:
  v87 = &v83;
  v88 = 0;
  v89 = &v81;
  v90 = (__int64 **)&v82;
  if ( !(unsigned __int8)sub_100AFE0((unsigned __int8 ***)&v87, (__int64)a1)
    || (v6 = *a2 == 59, v84 = 0, v85 = v81, v86 = v82, !v6)
    || (!(unsigned __int8)sub_995B10(&v84, *((_QWORD *)a2 - 8))
     || !(unsigned __int8)sub_1008380(&v85, *((_QWORD *)a2 - 4)))
    && (!(unsigned __int8)sub_995B10(&v84, *((_QWORD *)a2 - 4))
     || !(unsigned __int8)sub_1008380(&v85, *((_QWORD *)a2 - 8))) )
  {
    v6 = *(_BYTE *)a1 == 59;
    v87 = 0;
    v88 = &v81;
    v89 = &v82;
    v90 = &v84;
    if ( !v6 )
      return 0;
    v38 = sub_995B10(&v87, *(a1 - 8));
    v39 = (_BYTE *)*(a1 - 4);
    if ( v38 )
    {
      if ( *v39 == 59 )
      {
        v72 = *((_QWORD *)v39 - 8);
        if ( v72 )
        {
          *v88 = v72;
          v73 = *((_QWORD *)v39 - 4);
          if ( v73 )
          {
LABEL_172:
            *v89 = v73;
            *v90 = a1;
            if ( *a2 == 57 )
            {
              v74 = *((_QWORD *)a2 - 8);
              v75 = *((_QWORD *)a2 - 4);
              if ( v81 == v74 && v82 == v75 )
                return v84;
              if ( v82 == v74 && v81 == v75 )
                return v84;
            }
LABEL_83:
            v41 = *(_BYTE *)a1;
            v87 = 0;
            v88 = &v81;
            v89 = &v82;
            v90 = &v84;
            if ( v41 != 59 )
              return 0;
            v42 = sub_995B10(&v87, *(a1 - 8));
            v43 = (_BYTE *)*(a1 - 4);
            if ( v42 )
            {
              if ( *v43 == 57 )
              {
                v76 = *((_QWORD *)v43 - 8);
                if ( v76 )
                {
                  *v88 = v76;
                  v46 = *((_QWORD *)v43 - 4);
                  if ( v46 )
                  {
LABEL_90:
                    *v89 = v46;
                    *v90 = a1;
                    if ( *a2 != 59 )
                      return 0;
                    v47 = *((_QWORD *)a2 - 8);
                    v48 = *((_QWORD *)a2 - 4);
                    if ( (v81 != v47 || v82 != v48) && (v82 != v47 || v81 != v48) )
                      return 0;
                    return v84;
                  }
                  v43 = (_BYTE *)*(a1 - 4);
                }
              }
            }
            if ( !(unsigned __int8)sub_995B10(&v87, (__int64)v43) )
              return 0;
            v44 = (_BYTE *)*(a1 - 8);
            if ( *v44 != 57 )
              return 0;
            v45 = *((_QWORD *)v44 - 8);
            if ( !v45 )
              return 0;
            *v88 = v45;
            v46 = *((_QWORD *)v44 - 4);
            if ( !v46 )
              return 0;
            goto LABEL_90;
          }
          v39 = (_BYTE *)*(a1 - 4);
        }
      }
    }
    if ( !(unsigned __int8)sub_995B10(&v87, (__int64)v39) )
      goto LABEL_83;
    v40 = (_BYTE *)*(a1 - 8);
    if ( *v40 != 59 )
      goto LABEL_83;
    v77 = *((_QWORD *)v40 - 8);
    if ( !v77 )
      goto LABEL_83;
    *v88 = v77;
    v73 = *((_QWORD *)v40 - 4);
    if ( !v73 )
      goto LABEL_83;
    goto LABEL_172;
  }
  return (__int64 *)v83;
}
