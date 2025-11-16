// Function: sub_23D4B90
// Address: 0x23d4b90
//
__int64 __fastcall sub_23D4B90(__int64 a1, _QWORD *a2, _QWORD *a3, _QWORD *a4)
{
  __int64 v6; // r14
  __int64 v7; // rax
  _BYTE *v9; // rax
  _BYTE *v10; // rax
  char v11; // dl
  __int64 v12; // rdx
  _BYTE *v13; // rdx
  __int64 v14; // r15
  __int64 v15; // rax
  __int64 v16; // rax
  _BYTE *v17; // rax
  __int64 v18; // rdx
  _BYTE *v19; // rax
  _BYTE *v20; // rax
  char v21; // dl
  __int64 v22; // rdx
  __int64 v23; // rax
  __int64 v24; // rdx
  _BYTE *v25; // rdx
  __int64 v26; // rdx
  __int64 v27; // rax
  _BYTE *v28; // rax
  __int64 v29; // rdx
  _BYTE *v30; // rdx
  __int64 v31; // rdx
  _BYTE *v32; // rdx
  _BYTE *v33; // r8
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // rdx
  _BYTE *v37; // rax
  _BYTE *v38; // rax
  int v39; // eax
  int v40; // eax
  int v41; // [rsp+Ch] [rbp-74h]
  _BYTE *v42; // [rsp+10h] [rbp-70h]
  int v43; // [rsp+10h] [rbp-70h]
  __int64 v44; // [rsp+18h] [rbp-68h]
  __int64 v45; // [rsp+18h] [rbp-68h]
  _BYTE *v46; // [rsp+18h] [rbp-68h]
  _BYTE *v47; // [rsp+18h] [rbp-68h]
  _BYTE *v48; // [rsp+18h] [rbp-68h]
  _BYTE *v49; // [rsp+18h] [rbp-68h]
  __int64 v50; // [rsp+38h] [rbp-48h] BYREF
  _QWORD *v51; // [rsp+40h] [rbp-40h]

  v6 = (unsigned int)sub_BCB060(*(_QWORD *)(a1 + 8));
  v7 = *(_QWORD *)(a1 + 16);
  v50 = v6;
  v51 = a4;
  if ( !v7 )
    return 0;
  if ( *(_QWORD *)(v7 + 8) || *(_BYTE *)a1 != 58 )
  {
LABEL_4:
    if ( *(_QWORD *)(v7 + 8) || *(_BYTE *)a1 != 58 )
      return 0;
    v9 = *(_BYTE **)(a1 - 64);
    if ( *v9 != 54 )
      goto LABEL_7;
    v31 = *((_QWORD *)v9 - 8);
    if ( !v31 )
      goto LABEL_7;
    *a2 = v31;
    v32 = (_BYTE *)*((_QWORD *)v9 - 4);
    if ( *v32 != 44 )
      goto LABEL_7;
    v33 = (_BYTE *)*((_QWORD *)v32 - 8);
    if ( !v33 )
      goto LABEL_69;
    if ( *v33 != 17 )
    {
      if ( (unsigned int)*(unsigned __int8 *)(*((_QWORD *)v33 + 1) + 8LL) - 17 > 1 )
        goto LABEL_7;
      if ( *v33 > 0x15u )
        goto LABEL_7;
      v47 = (_BYTE *)*((_QWORD *)v9 - 4);
      v38 = sub_AD7630(*((_QWORD *)v32 - 8), 0, (__int64)v32);
      v32 = v47;
      v33 = v38;
      if ( !v38 || *v38 != 17 )
        goto LABEL_7;
    }
    if ( *((_DWORD *)v33 + 8) > 0x40u )
    {
      v41 = *((_DWORD *)v33 + 8);
      v42 = v32;
      v48 = v33;
      v39 = sub_C444A0((__int64)(v33 + 24));
      v32 = v42;
      if ( (unsigned int)(v41 - v39) > 0x40 )
        goto LABEL_7;
      v34 = **((_QWORD **)v48 + 3);
    }
    else
    {
      v34 = *((_QWORD *)v33 + 3);
    }
    if ( v6 == v34 )
    {
      v35 = *((_QWORD *)v32 - 4);
      if ( v35 )
      {
        *a4 = v35;
        v10 = *(_BYTE **)(a1 - 32);
        v11 = *v10;
        if ( *v10 != 55 )
          goto LABEL_8;
        v36 = *((_QWORD *)v10 - 8);
        if ( !v36 )
          return 0;
        *a3 = v36;
        if ( *((_QWORD *)v10 - 4) == *a4 )
          return 181;
      }
    }
LABEL_7:
    v10 = *(_BYTE **)(a1 - 32);
    v11 = *v10;
LABEL_8:
    if ( v11 != 54 )
      return 0;
    v12 = *((_QWORD *)v10 - 8);
    if ( !v12 )
      return 0;
    *a2 = v12;
    v13 = (_BYTE *)*((_QWORD *)v10 - 4);
    if ( *v13 != 44 )
      return 0;
    v14 = *((_QWORD *)v13 - 8);
    if ( v14 )
    {
      if ( *(_BYTE *)v14 != 17 )
      {
        v46 = (_BYTE *)*((_QWORD *)v10 - 4);
        if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v14 + 8) + 8LL) - 17 > 1 )
          return 0;
        if ( *(_BYTE *)v14 > 0x15u )
          return 0;
        v37 = sub_AD7630(v14, 0, (__int64)v13);
        v14 = (__int64)v37;
        if ( !v37 )
          return 0;
        v13 = v46;
        if ( *v37 != 17 )
          return 0;
      }
      if ( *(_DWORD *)(v14 + 32) <= 0x40u )
      {
        v15 = *(_QWORD *)(v14 + 24);
LABEL_15:
        if ( v6 != v15 )
          return 0;
        v16 = *((_QWORD *)v13 - 4);
        if ( !v16 )
          return 0;
        *a4 = v16;
        v17 = *(_BYTE **)(a1 - 64);
        if ( *v17 != 55 )
          return 0;
        v18 = *((_QWORD *)v17 - 8);
        if ( !v18 )
          return 0;
        *a3 = v18;
        if ( *((_QWORD *)v17 - 4) != *a4 )
          return 0;
        return 181;
      }
      v43 = *(_DWORD *)(v14 + 32);
      v49 = v13;
      v40 = sub_C444A0(v14 + 24);
      v13 = v49;
      if ( (unsigned int)(v43 - v40) <= 0x40 )
      {
        v15 = **(_QWORD **)(v14 + 24);
        goto LABEL_15;
      }
      return 0;
    }
LABEL_69:
    BUG();
  }
  v19 = *(_BYTE **)(a1 - 64);
  if ( *v19 == 54 )
  {
    v22 = *((_QWORD *)v19 - 8);
    if ( v22 )
    {
      *a2 = v22;
      v23 = *((_QWORD *)v19 - 4);
      if ( v23 )
      {
        *a4 = v23;
        v20 = *(_BYTE **)(a1 - 32);
        v21 = *v20;
        if ( *v20 != 55 )
          goto LABEL_24;
        v24 = *((_QWORD *)v20 - 8);
        if ( !v24 )
          goto LABEL_25;
        *a3 = v24;
        v25 = (_BYTE *)*((_QWORD *)v20 - 4);
        if ( *v25 == 44 )
        {
          v44 = *((_QWORD *)v20 - 4);
          if ( sub_F17ED0(&v50, *((_QWORD *)v25 - 8)) && *(_QWORD *)(v44 - 32) == *v51 )
            return 180;
        }
      }
    }
  }
  v20 = *(_BYTE **)(a1 - 32);
  v21 = *v20;
LABEL_24:
  if ( v21 != 54 )
    goto LABEL_25;
  v26 = *((_QWORD *)v20 - 8);
  if ( !v26 )
    goto LABEL_25;
  *a2 = v26;
  v27 = *((_QWORD *)v20 - 4);
  if ( !v27
    || (*a4 = v27, v28 = *(_BYTE **)(a1 - 64), *v28 != 55)
    || (v29 = *((_QWORD *)v28 - 8)) == 0
    || (*a3 = v29, v30 = (_BYTE *)*((_QWORD *)v28 - 4), *v30 != 44)
    || (v45 = *((_QWORD *)v28 - 4), !sub_F17ED0(&v50, *((_QWORD *)v30 - 8)))
    || *(_QWORD *)(v45 - 32) != *v51 )
  {
LABEL_25:
    v7 = *(_QWORD *)(a1 + 16);
    if ( !v7 )
      return 0;
    goto LABEL_4;
  }
  return 180;
}
