// Function: sub_100E540
// Address: 0x100e540
//
unsigned __int8 *__fastcall sub_100E540(__int64 *a1, _BYTE *a2, char a3, __m128i *a4, char a5, char a6)
{
  __int64 v10; // r8
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // rdi
  __int64 v15; // rax
  char v16; // al
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // rdi
  __int64 v20; // rax
  char v21; // al
  __int64 *v22; // rdx
  char v23; // al
  char v24; // al
  char v25; // al
  _BYTE *v26; // r13
  void *v27; // rax
  _BYTE *v28; // rax
  bool v29; // al
  __int64 v30; // r12
  __int64 v31; // rdx
  _BYTE *v32; // rax
  _BYTE *v33; // rbx
  void *v34; // rax
  _BYTE *v35; // rbx
  int v36; // ebx
  unsigned int v37; // r12d
  void **v38; // rax
  void **v39; // rdx
  char v40; // al
  void *v41; // rax
  _BYTE *v42; // rdx
  bool v43; // al
  bool v44; // [rsp+8h] [rbp-78h]
  __int64 v45; // [rsp+8h] [rbp-78h]
  __int64 v46; // [rsp+10h] [rbp-70h]
  void **v47; // [rsp+10h] [rbp-70h]
  bool v48; // [rsp+18h] [rbp-68h]
  __int64 v49; // [rsp+18h] [rbp-68h]
  _BYTE *v50; // [rsp+20h] [rbp-60h] BYREF
  __int64 *v51; // [rsp+28h] [rbp-58h] BYREF
  _QWORD v52[2]; // [rsp+30h] [rbp-50h] BYREF
  __int64 *v53; // [rsp+40h] [rbp-40h] BYREF
  _BYTE *v54; // [rsp+48h] [rbp-38h]

  v51 = a1;
  v50 = a2;
  v48 = a6 == 1 && a5 == 0;
  if ( v48 )
  {
    v10 = sub_FFE3E0(0xEu, (_BYTE **)&v51, &v50, a4->m128i_i64);
    if ( v10 )
      return (unsigned __int8 *)v10;
  }
  v53 = v51;
  v54 = v50;
  v10 = (__int64)sub_1003820((__int64 *)&v53, 2, a3, (__int64)a4, a5, a6);
  if ( v10 )
    return (unsigned __int8 *)v10;
  if ( (sub_9B4030(v51, 3, 0, a4) & 3) != 0 )
  {
    v14 = v51[1];
    if ( (unsigned int)*(unsigned __int8 *)(v14 + 8) - 17 <= 1 )
      v14 = **(_QWORD **)(v14 + 16);
    v15 = sub_BCAC60(v14, 3, v12, v13, 0);
    if ( sub_C33750(v15) )
      goto LABEL_9;
  }
  if ( !a5 || (a3 & 2) != 0 )
  {
    if ( (a6 & 0xFB) != 3 || (a3 & 8) != 0 )
    {
      v53 = 0;
      if ( (unsigned __int8)sub_1008640(&v53, (__int64)v50) )
        return (unsigned __int8 *)v51;
    }
LABEL_9:
    v16 = sub_9B4030(v51, 3, 0, a4);
    v10 = 0;
    if ( (v16 & 3) == 0 )
      goto LABEL_30;
    goto LABEL_10;
  }
  v25 = sub_9B4030(v51, 3, 0, a4);
  v10 = 0;
  if ( (v25 & 3) == 0 )
    return (unsigned __int8 *)v10;
LABEL_10:
  v19 = v51[1];
  if ( (unsigned int)*(unsigned __int8 *)(v19 + 8) - 17 <= 1 )
    v19 = **(_QWORD **)(v19 + 16);
  v20 = sub_BCAC60(v19, 3, v17, v18, 0);
  v21 = sub_C33750(v20);
  v10 = 0;
  if ( v21 )
    goto LABEL_13;
LABEL_30:
  if ( a5 && (a3 & 2) == 0 )
    return (unsigned __int8 *)v10;
  v53 = 0;
  v23 = sub_10069D0(&v53, (__int64)v50);
  v10 = 0;
  if ( v23 )
  {
    if ( (a3 & 8) != 0 )
      return (unsigned __int8 *)v51;
    v24 = sub_9B4030(v51, 32, 0, a4);
    v10 = 0;
    if ( (v24 & 0x20) == 0 )
      return (unsigned __int8 *)v51;
  }
LABEL_13:
  if ( !v48 )
    return (unsigned __int8 *)v10;
  if ( (a3 & 2) == 0 )
    goto LABEL_15;
  v26 = v50;
  if ( *v50 == 18 )
  {
    v27 = sub_C33340();
    v10 = 0;
    if ( *((void **)v26 + 3) == v27 )
      v28 = (_BYTE *)*((_QWORD *)v26 + 4);
    else
      v28 = v26 + 24;
    v29 = (v28[20] & 7) == 0;
    goto LABEL_43;
  }
  v30 = *((_QWORD *)v50 + 1);
  v31 = (unsigned int)*(unsigned __int8 *)(v30 + 8) - 17;
  if ( (unsigned int)v31 <= 1 && *v50 <= 0x15u )
  {
    v32 = sub_AD7630((__int64)v50, 0, v31);
    v10 = 0;
    v33 = v32;
    if ( v32 && *v32 == 18 )
    {
      v34 = sub_C33340();
      v10 = 0;
      if ( *((void **)v33 + 3) == v34 )
        v35 = (_BYTE *)*((_QWORD *)v33 + 4);
      else
        v35 = v33 + 24;
      v26 = v50;
      v29 = (v35[20] & 7) == 0;
LABEL_43:
      if ( v29 )
        return v26;
      goto LABEL_46;
    }
    if ( *(_BYTE *)(v30 + 8) == 17 )
    {
      v36 = *(_DWORD *)(v30 + 32);
      if ( v36 )
      {
        v44 = 0;
        v37 = 0;
        while ( 1 )
        {
          v46 = v10;
          v38 = (void **)sub_AD69F0(v26, v37);
          v10 = v46;
          v39 = v38;
          if ( !v38 )
            break;
          v40 = *(_BYTE *)v38;
          v47 = v39;
          if ( v40 != 13 )
          {
            if ( v40 != 18 )
              break;
            v45 = v10;
            v41 = sub_C33340();
            v10 = v45;
            v42 = v47[3] == v41 ? v47[4] : v47 + 3;
            if ( (v42[20] & 7) != 0 )
              break;
            v44 = v48;
          }
          if ( v36 == ++v37 )
          {
            v26 = v50;
            if ( v44 )
              return v26;
            goto LABEL_46;
          }
        }
      }
    }
    v26 = v50;
  }
LABEL_46:
  v49 = v10;
  v53 = 0;
  v54 = v26;
  if ( sub_100E3B0((__int64)&v53, 16, (unsigned __int8 *)v51) )
    return sub_AD9290(v51[1], 0);
  v52[0] = 0;
  v52[1] = v51;
  if ( sub_100E3B0((__int64)v52, 16, v50) )
    return sub_AD9290(v51[1], 0);
  v53 = (__int64 *)v50;
  if ( sub_1009310(&v53, (unsigned __int8 *)v51) )
    return sub_AD9290(v51[1], 0);
  v52[0] = v51;
  v43 = sub_1009310(v52, v50);
  v10 = v49;
  if ( v43 )
    return sub_AD9290(v51[1], 0);
LABEL_15:
  if ( (a3 & 8) != 0 && (a3 & 1) != 0 )
  {
    if ( *(_BYTE *)v51 == 45 && *(v51 - 8) && v50 == (_BYTE *)*(v51 - 4) )
    {
      return (unsigned __int8 *)*(v51 - 8);
    }
    else if ( *v50 == 45 )
    {
      if ( *((_QWORD *)v50 - 8) )
      {
        v22 = (__int64 *)*((_QWORD *)v50 - 4);
        if ( v22 )
        {
          if ( v51 == v22 )
            return (unsigned __int8 *)*((_QWORD *)v50 - 8);
        }
      }
    }
  }
  return (unsigned __int8 *)v10;
}
