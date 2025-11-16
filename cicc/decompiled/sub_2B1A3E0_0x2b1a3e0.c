// Function: sub_2B1A3E0
// Address: 0x2b1a3e0
//
_QWORD *__fastcall sub_2B1A3E0(_QWORD *a1, __int64 a2, char **a3)
{
  _QWORD *result; // rax
  __int64 v6; // r8
  __int64 v7; // rdx
  _QWORD *v8; // r8
  _BYTE *v9; // rcx
  __int64 v10; // rdx
  char v11; // si
  __int64 v12; // rcx
  __int64 v13; // rdx
  char v14; // dl
  char *v15; // rdi
  _QWORD *v16; // rsi
  __int64 v17; // rcx
  char v18; // cl
  __int64 v19; // r11
  __int64 v20; // rdi
  char *v21; // rdi
  __int64 v22; // rcx
  char v23; // cl
  __int64 v24; // r11
  __int64 v25; // rdi
  char *v26; // rdi
  __int64 v27; // rcx
  char v28; // cl
  __int64 v29; // r11
  __int64 v30; // rdi
  _BYTE *v31; // rdx
  __int64 v32; // rcx
  char v33; // cl
  __int64 v34; // rdx
  __int64 v35; // rsi
  _BYTE *v36; // rdx
  __int64 v37; // rcx
  char v38; // cl
  __int64 v39; // rdx
  __int64 v40; // rsi
  _BYTE *v41; // rdx
  __int64 v42; // rcx
  char v43; // cl
  __int64 v44; // rdx
  __int64 v45; // rsi

  result = a1;
  v6 = (a2 - (__int64)a1) >> 5;
  v7 = (a2 - (__int64)a1) >> 3;
  if ( v6 <= 0 )
  {
LABEL_43:
    if ( v7 != 2 )
    {
      if ( v7 != 3 )
      {
        if ( v7 != 1 )
          return (_QWORD *)a2;
LABEL_67:
        v41 = (_BYTE *)*result;
        v42 = *(_QWORD *)(*result + 16LL);
        if ( v42 )
        {
          if ( !*(_QWORD *)(v42 + 8) )
          {
            v43 = *v41;
            if ( *v41 > 0x1Cu && (v43 == 68 || v43 == 69) )
            {
              v44 = *((_QWORD *)v41 - 4);
              v45 = *(_QWORD *)(v44 + 16);
              if ( v45 && !*(_QWORD *)(v45 + 8) && *(_BYTE *)v44 == 61 && **a3 == v43 )
                return (_QWORD *)a2;
            }
          }
        }
        return result;
      }
      v31 = (_BYTE *)*result;
      v32 = *(_QWORD *)(*result + 16LL);
      if ( !v32 )
        return result;
      if ( *(_QWORD *)(v32 + 8) )
        return result;
      v33 = *v31;
      if ( *v31 <= 0x1Cu || v33 != 68 && v33 != 69 )
        return result;
      v34 = *((_QWORD *)v31 - 4);
      v35 = *(_QWORD *)(v34 + 16);
      if ( !v35 || *(_QWORD *)(v35 + 8) || *(_BYTE *)v34 != 61 || **a3 != v33 )
        return result;
      ++result;
    }
    v36 = (_BYTE *)*result;
    v37 = *(_QWORD *)(*result + 16LL);
    if ( !v37 )
      return result;
    if ( *(_QWORD *)(v37 + 8) )
      return result;
    v38 = *v36;
    if ( *v36 <= 0x1Cu || v38 != 68 && v38 != 69 )
      return result;
    v39 = *((_QWORD *)v36 - 4);
    v40 = *(_QWORD *)(v39 + 16);
    if ( !v40 || *(_QWORD *)(v40 + 8) || *(_BYTE *)v39 != 61 || **a3 != v38 )
      return result;
    ++result;
    goto LABEL_67;
  }
  v8 = &a1[4 * v6];
  while ( 1 )
  {
    v9 = (_BYTE *)*result;
    v10 = *(_QWORD *)(*result + 16LL);
    if ( !v10 )
      return result;
    if ( *(_QWORD *)(v10 + 8) )
      return result;
    v11 = *v9;
    if ( *v9 <= 0x1Cu || v11 != 68 && v11 != 69 )
      return result;
    v12 = *((_QWORD *)v9 - 4);
    v13 = *(_QWORD *)(v12 + 16);
    if ( !v13 )
      return result;
    if ( *(_QWORD *)(v13 + 8) )
      return result;
    if ( *(_BYTE *)v12 != 61 )
      return result;
    v14 = **a3;
    if ( v14 != v11 )
      return result;
    v15 = (char *)result[1];
    v16 = result + 1;
    v17 = *((_QWORD *)v15 + 2);
    if ( !v17 )
      return v16;
    if ( *(_QWORD *)(v17 + 8) )
      return v16;
    v18 = *v15;
    if ( (unsigned __int8)*v15 <= 0x1Cu || v18 != 68 && v18 != 69 )
      return v16;
    v19 = *((_QWORD *)v15 - 4);
    v20 = *(_QWORD *)(v19 + 16);
    if ( !v20 )
      return v16;
    if ( *(_QWORD *)(v20 + 8) )
      return v16;
    if ( *(_BYTE *)v19 != 61 )
      return v16;
    if ( v14 != v18 )
      return v16;
    v21 = (char *)result[2];
    v16 = result + 2;
    v22 = *((_QWORD *)v21 + 2);
    if ( !v22 )
      return v16;
    if ( *(_QWORD *)(v22 + 8) )
      return v16;
    v23 = *v21;
    if ( (unsigned __int8)*v21 <= 0x1Cu || v23 != 68 && v23 != 69 )
      return v16;
    v24 = *((_QWORD *)v21 - 4);
    v25 = *(_QWORD *)(v24 + 16);
    if ( !v25 )
      return v16;
    if ( *(_QWORD *)(v25 + 8) )
      return v16;
    if ( *(_BYTE *)v24 != 61 )
      return v16;
    if ( v14 != v23 )
      return v16;
    v26 = (char *)result[3];
    v16 = result + 3;
    v27 = *((_QWORD *)v26 + 2);
    if ( !v27 )
      return v16;
    if ( *(_QWORD *)(v27 + 8) )
      return v16;
    v28 = *v26;
    if ( (unsigned __int8)*v26 <= 0x1Cu || v28 != 68 && v28 != 69 )
      return v16;
    v29 = *((_QWORD *)v26 - 4);
    v30 = *(_QWORD *)(v29 + 16);
    if ( !v30 || *(_QWORD *)(v30 + 8) || *(_BYTE *)v29 != 61 || v14 != v28 )
      return v16;
    result += 4;
    if ( v8 == result )
    {
      v7 = (a2 - (__int64)result) >> 3;
      goto LABEL_43;
    }
  }
}
