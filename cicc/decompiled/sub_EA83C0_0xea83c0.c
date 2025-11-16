// Function: sub_EA83C0
// Address: 0xea83c0
//
__int64 __fastcall sub_EA83C0(const char *a1, __int64 a2, char a3, _QWORD *a4, __int64 *a5, __int64 *a6)
{
  __int64 v10; // rax
  unsigned int v11; // r15d
  __int64 (__fastcall *v13)(__int64); // rax
  __int64 v14; // rdi
  __int64 v15; // rax
  const char *v16; // rax
  __int64 v17; // rax
  __int64 (__fastcall *v18)(__int64); // rax
  __int64 v19; // rdi
  __int64 v20; // rax
  __int64 (__fastcall *v21)(__int64); // rax
  __int64 v22; // rdi
  __int64 v23; // rbx
  __int64 v24; // rax
  void *v25; // rax
  char v26; // si
  char v27; // cl
  _BYTE *v28; // rax
  void *v29; // rax
  __int64 v31; // [rsp+10h] [rbp-A0h]
  _QWORD v33[4]; // [rsp+20h] [rbp-90h] BYREF
  __int16 v34; // [rsp+40h] [rbp-70h]
  const char *v35; // [rsp+50h] [rbp-60h] BYREF
  __int64 v36; // [rsp+58h] [rbp-58h]
  char *v37; // [rsp+60h] [rbp-50h]
  __int16 v38; // [rsp+70h] [rbp-40h]

  v10 = sub_ECD7B0(a4);
  v31 = sub_ECD6A0(v10);
  if ( (unsigned __int8)sub_ECD870(a4, a6) )
  {
    v35 = "missing expression";
    v38 = 259;
    return (unsigned int)sub_ECE0E0(a4, &v35, 0, 0);
  }
  v11 = sub_ECE000(a4);
  if ( (_BYTE)v11 )
    return v11;
  v13 = *(__int64 (__fastcall **)(__int64))(*a4 + 48LL);
  if ( v13 == sub_EA2270 )
    v14 = a4[28];
  else
    v14 = v13((__int64)a4);
  v35 = a1;
  v38 = 261;
  v36 = a2;
  v15 = sub_E65280(v14, &v35);
  *a5 = v15;
  if ( v15 )
  {
    if ( sub_E806E0(*a6, v15) )
    {
      v16 = "Recursive use of '";
      v34 = 1283;
LABEL_10:
      v33[2] = a1;
      v33[0] = v16;
      v35 = (const char *)v33;
      v38 = 770;
      v33[3] = a2;
      v37 = "'";
      return (unsigned int)sub_ECDA70(a4, v31, &v35, 0, 0);
    }
    v23 = *a5;
    v20 = *a5;
    if ( *(_QWORD *)*a5 )
      goto LABEL_21;
    if ( (*(_BYTE *)(v23 + 9) & 0x70) != 0x20 )
    {
      if ( (*(_BYTE *)(v23 + 8) & 8) == 0 )
        goto LABEL_16;
      goto LABEL_43;
    }
    if ( *(char *)(v23 + 8) < 0 )
    {
      if ( (*(_BYTE *)(v23 + 8) & 8) != 0 )
      {
LABEL_46:
        if ( *(char *)(v23 + 8) < 0 )
        {
LABEL_37:
          v28 = *(_BYTE **)(v23 + 24);
          *(_BYTE *)(v23 + 8) |= 8u;
          if ( *v28 != 1 )
          {
            v16 = "invalid reassignment of non-absolute variable '";
            v34 = 1283;
            goto LABEL_10;
          }
          v20 = *a5;
LABEL_16:
          *(_BYTE *)(v20 + 8) = (4 * (a3 & 1)) | *(_BYTE *)(v20 + 8) & 0xFB;
          return v11;
        }
        *(_BYTE *)(v23 + 8) |= 8u;
        v29 = sub_E807D0(*(_QWORD *)(v23 + 24));
        *(_QWORD *)v23 = v29;
        v23 = *a5;
        if ( !v29 )
        {
LABEL_43:
          if ( (*(_BYTE *)(v23 + 9) & 0x70) != 0x20 )
          {
            v16 = "invalid assignment to '";
            v34 = 1283;
            goto LABEL_10;
          }
          goto LABEL_37;
        }
LABEL_35:
        if ( (*(_BYTE *)(v23 + 9) & 0x70) != 0x20 || !a3 )
          goto LABEL_24;
        goto LABEL_37;
      }
      goto LABEL_32;
    }
    v25 = sub_E807D0(*(_QWORD *)(v23 + 24));
    *(_QWORD *)v23 = v25;
    v23 = *a5;
    if ( v25 )
    {
LABEL_21:
      if ( (*(_BYTE *)(v23 + 9) & 0x70) != 0x20 )
      {
        v24 = *(_QWORD *)v23;
        goto LABEL_23;
      }
      v27 = *(_BYTE *)(v23 + 8) & 8;
    }
    else
    {
      v20 = *a5;
      v26 = *(_BYTE *)(v23 + 9) & 0x70;
      v27 = *(_BYTE *)(v23 + 8) & 8;
      if ( v27 )
      {
        v24 = *(_QWORD *)v23;
        if ( v26 != 32 )
        {
LABEL_23:
          if ( v24 )
          {
LABEL_24:
            v16 = "redefinition of '";
            v34 = 1283;
            goto LABEL_10;
          }
          goto LABEL_43;
        }
LABEL_34:
        if ( v24 )
          goto LABEL_35;
        goto LABEL_46;
      }
      if ( v26 != 32 )
        goto LABEL_16;
    }
    if ( v27 )
    {
LABEL_33:
      v24 = *(_QWORD *)v23;
      goto LABEL_34;
    }
LABEL_32:
    v20 = v23;
    if ( a3 )
      goto LABEL_16;
    goto LABEL_33;
  }
  v17 = *a4;
  if ( a2 != 1 || *a1 != 46 )
  {
    v18 = *(__int64 (__fastcall **)(__int64))(v17 + 48);
    if ( v18 == sub_EA2270 )
      v19 = a4[28];
    else
      v19 = v18((__int64)a4);
    v35 = a1;
    v38 = 261;
    v36 = a2;
    v20 = sub_E6C460(v19, &v35);
    *a5 = v20;
    goto LABEL_16;
  }
  v21 = *(__int64 (__fastcall **)(__int64))(v17 + 56);
  if ( v21 == sub_EA2280 )
    v22 = a4[29];
  else
    v22 = v21((__int64)a4);
  (*(void (__fastcall **)(__int64, __int64, _QWORD, __int64))(*(_QWORD *)v22 + 624LL))(v22, *a6, 0, v31);
  return v11;
}
