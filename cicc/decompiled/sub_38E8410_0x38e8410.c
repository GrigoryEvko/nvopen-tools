// Function: sub_38E8410
// Address: 0x38e8410
//
__int64 __fastcall sub_38E8410(_BYTE *a1, __int64 a2, char a3, _QWORD *a4, __int64 *a5, __int64 *a6)
{
  __int64 v9; // rax
  __int64 v10; // r8
  __int64 v11; // r9
  unsigned int v12; // r15d
  __int64 (__fastcall *v14)(__int64); // rax
  __int64 v15; // rdi
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 (__fastcall *v18)(__int64); // rax
  __int64 v19; // rdi
  __int64 v20; // rax
  __int64 v21; // rbx
  const char *v22; // rax
  unsigned __int64 v23; // rax
  __int64 (__fastcall *v24)(__int64); // rax
  __int64 v25; // rdi
  char v26; // cl
  unsigned __int64 v27; // rax
  _DWORD *v28; // rax
  unsigned __int64 v29; // rax
  char v30; // si
  __int64 v31; // [rsp+10h] [rbp-90h]
  _BYTE *v33; // [rsp+20h] [rbp-80h] BYREF
  __int64 v34; // [rsp+28h] [rbp-78h]
  _QWORD v35[2]; // [rsp+30h] [rbp-70h] BYREF
  __int16 v36; // [rsp+40h] [rbp-60h]
  _QWORD v37[2]; // [rsp+50h] [rbp-50h] BYREF
  __int16 v38; // [rsp+60h] [rbp-40h]

  v33 = a1;
  v34 = a2;
  v9 = sub_3909460(a4);
  v31 = sub_39092A0(v9);
  HIBYTE(v38) = 1;
  if ( (unsigned __int8)sub_3909510(a4, a6) )
  {
    v37[0] = "missing expression";
    LOBYTE(v38) = 3;
    return (unsigned int)sub_3909CF0(a4, v37, 0, 0, v10, v11);
  }
  LOBYTE(v38) = 3;
  v37[0] = "unexpected token";
  v12 = sub_3909E20(a4, 9, v37);
  if ( (_BYTE)v12 )
    return v12;
  v14 = *(__int64 (__fastcall **)(__int64))(*a4 + 48LL);
  if ( v14 == sub_38E2A70 )
    v15 = a4[40];
  else
    v15 = v14((__int64)a4);
  v38 = 261;
  v37[0] = &v33;
  v16 = sub_38BD730(v15, (__int64)v37);
  *a5 = v16;
  if ( v16 )
  {
    if ( sub_38E2DE0(v16, *a6) )
    {
      v36 = 1283;
      v35[0] = "Recursive use of '";
LABEL_10:
      v35[1] = &v33;
      v37[0] = v35;
      v38 = 770;
      v37[1] = "'";
      return (unsigned int)sub_3909790(a4, v31, v37, 0, 0);
    }
    v21 = *a5;
    v20 = *a5;
    if ( (*(_QWORD *)*a5 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
      goto LABEL_23;
    if ( (*(_BYTE *)(v21 + 9) & 0xC) != 8 )
    {
      if ( (*(_BYTE *)(v21 + 8) & 4) == 0 )
        goto LABEL_15;
      goto LABEL_19;
    }
    v29 = (unsigned __int64)sub_38CE440(*(_QWORD *)(v21 + 24));
    *(_QWORD *)v21 = v29 | *(_QWORD *)v21 & 7LL;
    v21 = *a5;
    if ( v29 )
    {
LABEL_23:
      if ( (*(_BYTE *)(v21 + 9) & 0xC) == 8 )
      {
        v26 = *(_BYTE *)(v21 + 8) & 4;
LABEL_32:
        if ( !v26 )
        {
          v20 = v21;
          if ( a3 )
            goto LABEL_15;
        }
        v23 = *(_QWORD *)v21 & 0xFFFFFFFFFFFFFFF8LL;
LABEL_35:
        if ( v23
          || (*(_BYTE *)(v21 + 8) |= 4u,
              v27 = (unsigned __int64)sub_38CE440(*(_QWORD *)(v21 + 24)),
              *(_QWORD *)v21 = v27 | *(_QWORD *)v21 & 7LL,
              v21 = *a5,
              v27) )
        {
          if ( (*(_BYTE *)(v21 + 9) & 0xC) == 8 && a3 )
          {
LABEL_39:
            v28 = *(_DWORD **)(v21 + 24);
            *(_BYTE *)(v21 + 8) |= 4u;
            if ( *v28 != 1 )
            {
              v22 = "invalid reassignment of non-absolute variable '";
              v36 = 1283;
              goto LABEL_21;
            }
            v20 = *a5;
LABEL_15:
            *(_BYTE *)(v20 + 8) = (2 * (a3 & 1)) | *(_BYTE *)(v20 + 8) & 0xFD;
            return v12;
          }
LABEL_26:
          v22 = "redefinition of '";
          v36 = 1283;
          goto LABEL_21;
        }
LABEL_19:
        if ( (*(_BYTE *)(v21 + 9) & 0xC) != 8 )
        {
          v22 = "invalid assignment to '";
          v36 = 1283;
LABEL_21:
          v35[0] = v22;
          goto LABEL_10;
        }
        goto LABEL_39;
      }
      v23 = *(_QWORD *)v21 & 0xFFFFFFFFFFFFFFF8LL;
    }
    else
    {
      v20 = *a5;
      v30 = *(_BYTE *)(v21 + 9) & 0xC;
      v26 = *(_BYTE *)(v21 + 8) & 4;
      if ( !v26 )
      {
        if ( v30 != 8 )
          goto LABEL_15;
        goto LABEL_32;
      }
      v23 = *(_QWORD *)v21 & 0xFFFFFFFFFFFFFFF8LL;
      if ( v30 == 8 )
        goto LABEL_35;
    }
    if ( v23 )
      goto LABEL_26;
    goto LABEL_19;
  }
  v17 = *a4;
  if ( v34 != 1 || *v33 != 46 )
  {
    v18 = *(__int64 (__fastcall **)(__int64))(v17 + 48);
    if ( v18 == sub_38E2A70 )
      v19 = a4[40];
    else
      v19 = v18((__int64)a4);
    v38 = 261;
    v37[0] = &v33;
    v20 = sub_38BF510(v19, (__int64)v37);
    *a5 = v20;
    goto LABEL_15;
  }
  v24 = *(__int64 (__fastcall **)(__int64))(v17 + 56);
  if ( v24 == sub_38E2A80 )
    v25 = a4[41];
  else
    v25 = v24((__int64)a4);
  (*(void (__fastcall **)(__int64, __int64, _QWORD, __int64))(*(_QWORD *)v25 + 528LL))(v25, *a6, 0, v31);
  return v12;
}
