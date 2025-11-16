// Function: sub_101B9B0
// Address: 0x101b9b0
//
unsigned __int8 *__fastcall sub_101B9B0(
        unsigned __int8 *a1,
        unsigned __int8 *a2,
        char a3,
        char a4,
        __m128i *a5,
        int a6)
{
  unsigned __int8 *v6; // r13
  unsigned __int8 v9; // al
  unsigned __int8 *v10; // r14
  bool v12; // dl
  unsigned __int8 v13; // al
  unsigned __int8 v14; // cl
  __int64 v15; // r15
  char v16; // al
  __int64 v17; // rsi
  char v18; // al
  __int64 v19; // rdi
  char v20; // al
  __int64 v21; // rsi
  char v22; // al
  __int64 v23; // rcx
  unsigned __int8 *v24; // rsi
  unsigned __int8 *v25; // rsi
  char v26; // al
  _QWORD **v27; // r8
  __int64 v28; // rdx
  __int64 v29; // rax
  __int64 *v30; // rcx
  __int64 v31; // rsi
  unsigned __int8 v32; // al
  unsigned __int8 v33; // dl
  __int64 v34; // r15
  __int64 *v35; // rax
  unsigned __int8 *v36; // rcx
  unsigned __int8 v37; // al
  int v38; // eax
  unsigned int v39; // r15d
  __int64 v40; // rax
  unsigned __int8 v41; // al
  _QWORD **v42; // [rsp+0h] [rbp-80h]
  bool v43; // [rsp+8h] [rbp-78h]
  unsigned __int8 v44; // [rsp+8h] [rbp-78h]
  bool v45; // [rsp+10h] [rbp-70h]
  __int64 v46; // [rsp+10h] [rbp-70h]
  bool v47; // [rsp+18h] [rbp-68h]
  __int64 v48; // [rsp+18h] [rbp-68h]
  int v49; // [rsp+18h] [rbp-68h]
  __int64 v51; // [rsp+20h] [rbp-60h]
  _QWORD *v53; // [rsp+30h] [rbp-50h] BYREF
  unsigned __int8 *v54; // [rsp+38h] [rbp-48h]
  _QWORD *v55; // [rsp+40h] [rbp-40h] BYREF
  unsigned __int8 *v56; // [rsp+48h] [rbp-38h]

  v6 = a1;
  v9 = *a2;
  if ( *a1 > 0x15u )
  {
    v10 = a2;
  }
  else if ( v9 > 0x15u )
  {
    v9 = *a1;
    v10 = a1;
    v6 = a2;
  }
  else
  {
    v10 = (unsigned __int8 *)sub_96E6C0(0xDu, (__int64)a1, a2, a5->m128i_i64[0]);
    if ( v10 )
      return v10;
    v9 = *a2;
    v10 = a2;
  }
  if ( v9 == 13 || (unsigned __int8)sub_1003090((__int64)a5, v10) )
    return v10;
  if ( (unsigned __int8)sub_FFFE90((__int64)v10) )
    return v6;
  v12 = sub_98F660(v6, v10, 0, 1);
  if ( v12 )
    return (unsigned __int8 *)sub_AD6530(*((_QWORD *)v6 + 1), (__int64)v10);
  v13 = *v10;
  if ( *v10 == 44 )
  {
    v23 = *((_QWORD *)v10 - 8);
    if ( v23 )
    {
      v24 = (unsigned __int8 *)*((_QWORD *)v10 - 4);
      if ( v24 )
      {
        if ( v6 == v24 )
          return (unsigned __int8 *)v23;
      }
    }
  }
  v14 = *v6;
  if ( *v6 == 44 )
  {
    v23 = *((_QWORD *)v6 - 8);
    if ( v23 )
    {
      v25 = (unsigned __int8 *)*((_QWORD *)v6 - 4);
      if ( v25 )
      {
        if ( v10 == v25 )
          return (unsigned __int8 *)v23;
      }
    }
    v53 = 0;
    v15 = *((_QWORD *)v6 + 1);
    v54 = v10;
  }
  else
  {
    v53 = 0;
    v15 = *((_QWORD *)v6 + 1);
    v54 = v10;
    if ( v14 == 59 )
    {
      v16 = sub_995B10(&v53, *((_QWORD *)v6 - 8));
      v17 = *((_QWORD *)v6 - 4);
      if ( v16 && (unsigned __int8 *)v17 == v54 )
        return (unsigned __int8 *)sub_AD62B0(v15);
      v18 = sub_995B10(&v53, v17);
      v12 = 0;
      if ( v18 )
      {
        if ( *((unsigned __int8 **)v6 - 8) == v54 )
          return (unsigned __int8 *)sub_AD62B0(v15);
      }
      v13 = *v10;
    }
  }
  v55 = 0;
  v56 = v6;
  if ( v13 == 59 )
  {
    v45 = v12;
    v20 = sub_995B10(&v55, *((_QWORD *)v10 - 8));
    v21 = *((_QWORD *)v10 - 4);
    if ( v20 )
    {
      if ( (unsigned __int8 *)v21 == v56 )
        return (unsigned __int8 *)sub_AD62B0(v15);
    }
    v22 = sub_995B10(&v55, v21);
    v12 = v45;
    if ( v22 )
    {
      if ( *((unsigned __int8 **)v10 - 8) == v56 )
        return (unsigned __int8 *)sub_AD62B0(v15);
    }
  }
  if ( a3 || a4 )
  {
    v47 = v12;
    v55 = 0;
    v26 = sub_993BE0(&v55, (__int64)v10);
    v27 = &v55;
    if ( v26 )
    {
      v28 = v47;
      if ( *v6 == 59 )
      {
        v51 = *((_QWORD *)v6 - 8);
        if ( v51 )
        {
          v30 = (__int64 *)*((_QWORD *)v6 - 4);
          v31 = *(unsigned __int8 *)v30;
          if ( (_BYTE)v31 == 17 )
          {
            v32 = sub_986B30(v30 + 3, v31, v47, (__int64)v30, (unsigned int)&v55);
            v27 = &v55;
            v33 = v32;
            goto LABEL_53;
          }
          v34 = v30[1];
          v43 = v47;
          if ( (unsigned int)*(unsigned __int8 *)(v34 + 8) - 17 <= 1 && (unsigned __int8)v31 <= 0x15u )
          {
            v48 = *((_QWORD *)v6 - 4);
            v35 = (__int64 *)sub_AD7630((__int64)v30, 0, v28);
            v36 = (unsigned __int8 *)v48;
            v27 = &v55;
            v33 = v43;
            if ( v35 && *(_BYTE *)v35 == 17 )
            {
              v37 = sub_986B30(v35 + 3, 0, v43, v48, (unsigned int)&v55);
              v27 = &v55;
              v33 = v37;
              goto LABEL_53;
            }
            if ( *(_BYTE *)(v34 + 8) == 17 )
            {
              v38 = *(_DWORD *)(v34 + 32);
              v39 = 0;
              v49 = v38;
              while ( v49 != v39 )
              {
                v42 = v27;
                v44 = v33;
                v46 = (__int64)v36;
                v40 = sub_AD69F0(v36, v39);
                v27 = v42;
                if ( !v40 )
                  goto LABEL_45;
                v36 = (unsigned __int8 *)v46;
                v33 = v44;
                if ( *(_BYTE *)v40 != 13 )
                {
                  if ( *(_BYTE *)v40 != 17 )
                    goto LABEL_45;
                  v41 = sub_986B30((__int64 *)(v40 + 24), v39, v44, v46, (unsigned int)v42);
                  v27 = v42;
                  v36 = (unsigned __int8 *)v46;
                  v33 = v41;
                  if ( !v41 )
                    goto LABEL_45;
                }
                ++v39;
              }
LABEL_53:
              if ( v33 )
                return (unsigned __int8 *)v51;
            }
          }
        }
      }
    }
LABEL_45:
    if ( a4 )
    {
      v55 = 0;
      if ( (unsigned __int8)sub_995B10(v27, (__int64)v10) )
        return v10;
    }
  }
  if ( a6 )
  {
    v19 = *((_QWORD *)v6 + 1);
    if ( (unsigned int)*(unsigned __int8 *)(v19 + 8) - 17 <= 1 )
      v19 = **(_QWORD **)(v19 + 16);
    if ( sub_BCAC40(v19, 1) )
    {
      v29 = sub_101B6D0(v6, v10, a5, a6 - 1);
      if ( v29 )
        return (unsigned __int8 *)v29;
    }
  }
  return sub_101B370(13, (__int64 *)v6, (__int64 *)v10, a5, a6);
}
