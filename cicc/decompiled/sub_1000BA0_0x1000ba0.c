// Function: sub_1000BA0
// Address: 0x1000ba0
//
__int64 __fastcall sub_1000BA0(__int64 a1, __int64 a2, char *a3)
{
  _BYTE *v4; // rax
  __int64 result; // rax
  __int64 v6; // rdi
  __int64 v8; // r12
  __int64 v9; // rdi
  int v10; // ebx
  int v11; // eax
  __int64 v12; // rdx
  char v13; // r13
  char v14; // r13
  unsigned int v15; // esi
  unsigned int v16; // r15d
  _QWORD *v17; // r14
  __int64 v18; // rax
  bool v19; // al
  __int64 v20; // rdx
  _BYTE *v21; // rax
  bool v22; // al
  _QWORD *v23; // rax
  __int64 v24; // rdx
  _BYTE *v25; // rax
  __int64 v26; // rax
  __int64 v27; // [rsp-80h] [rbp-80h]
  _BYTE *v28; // [rsp-78h] [rbp-78h]
  __int64 v29; // [rsp-78h] [rbp-78h]
  bool v30; // [rsp-6Ch] [rbp-6Ch]
  __int64 v31; // [rsp-68h] [rbp-68h]
  int v32; // [rsp-68h] [rbp-68h]
  __int64 v33; // [rsp-68h] [rbp-68h]
  _QWORD *v34; // [rsp-58h] [rbp-58h] BYREF
  unsigned int v35; // [rsp-50h] [rbp-50h]
  _QWORD *v36; // [rsp-48h] [rbp-48h] BYREF
  unsigned int v37; // [rsp-40h] [rbp-40h]

  if ( !a1 )
    return 0;
  v4 = *(_BYTE **)(a1 - 64);
  if ( *v4 != 42 )
    return 0;
  v31 = *((_QWORD *)v4 - 8);
  if ( !v31 )
    return 0;
  v6 = *((_QWORD *)v4 - 4);
  v8 = v6 + 24;
  if ( *(_BYTE *)v6 != 17 )
  {
    v20 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v6 + 8) + 8LL) - 17;
    if ( (unsigned int)v20 > 1 )
      return 0;
    if ( *(_BYTE *)v6 > 0x15u )
      return 0;
    v21 = sub_AD7630(v6, 0, v20);
    if ( !v21 || *v21 != 17 )
      return 0;
    v8 = (__int64)(v21 + 24);
  }
  v9 = *(_QWORD *)(a1 - 32);
  if ( *(_BYTE *)v9 == 17 )
  {
    v28 = (_BYTE *)(v9 + 24);
  }
  else
  {
    v24 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v9 + 8) + 8LL) - 17;
    if ( (unsigned int)v24 > 1 )
      return 0;
    if ( *(_BYTE *)v9 > 0x15u )
      return 0;
    v25 = sub_AD7630(v9, 0, v24);
    if ( !v25 || *v25 != 17 )
      return 0;
    v28 = v25 + 24;
  }
  v10 = sub_B53900(a1);
  if ( !a2 || v31 != *(_QWORD *)(a2 - 64) )
    return 0;
  v11 = sub_B53900(a2);
  v12 = *(_QWORD *)(a1 - 64);
  v32 = v11;
  result = 0;
  if ( *(_QWORD *)(v12 - 32) != *(_QWORD *)(a2 - 32) )
    return result;
  v13 = *a3;
  v30 = 0;
  v27 = *(_QWORD *)(a1 + 8);
  if ( v13 )
  {
    v14 = *(_BYTE *)(v12 + 1);
    v30 = (v14 & 4) != 0;
    v13 = (v14 & 2) != 0;
  }
  v37 = *((_DWORD *)v28 + 2);
  if ( v37 > 0x40 )
    sub_C43780((__int64)&v36, (const void **)v28);
  else
    v36 = *(_QWORD **)v28;
  sub_C46B40((__int64)&v36, (__int64 *)v8);
  v15 = *(_DWORD *)(v8 + 8);
  v16 = v37;
  v17 = v36;
  v35 = v37;
  v34 = v36;
  v18 = 1LL << ((unsigned __int8)v15 - 1);
  if ( v15 > 0x40 )
  {
    v29 = *(_QWORD *)(*(_QWORD *)v8 + 8LL * ((v15 - 1) >> 6)) & v18;
    v22 = v15 == (unsigned int)sub_C444A0(v8);
    if ( v29 )
      goto LABEL_44;
LABEL_38:
    if ( v22 )
      goto LABEL_43;
    if ( v16 > 0x40 )
    {
      if ( v16 - (unsigned int)sub_C444A0((__int64)&v34) > 0x40 )
        goto LABEL_43;
      v23 = (_QWORD *)*v17;
      if ( *v17 != 2 )
        goto LABEL_42;
    }
    else if ( v17 != (_QWORD *)2 )
    {
      goto LABEL_41;
    }
    if ( v10 == 36 )
    {
      if ( v32 == 38 )
        goto LABEL_31;
    }
    else if ( v10 == 40 && v32 == 38 && v30 )
    {
      goto LABEL_31;
    }
    if ( v16 > 0x40 )
    {
      if ( v16 - (unsigned int)sub_C444A0((__int64)&v34) > 0x40 )
        goto LABEL_43;
      v23 = (_QWORD *)*v17;
LABEL_42:
      if ( v23 == (_QWORD *)1 )
      {
        if ( v10 == 37 )
        {
          if ( v32 == 38 )
            goto LABEL_31;
        }
        else if ( v10 == 41 && v32 == 38 && v30 )
        {
          goto LABEL_31;
        }
      }
LABEL_43:
      if ( v15 <= 0x40 )
        goto LABEL_18;
LABEL_44:
      v19 = v15 == (unsigned int)sub_C444A0(v8);
      goto LABEL_19;
    }
LABEL_41:
    v23 = v17;
    goto LABEL_42;
  }
  if ( (*(_QWORD *)v8 & v18) == 0 )
  {
    v22 = *(_QWORD *)v8 == 0;
    goto LABEL_38;
  }
LABEL_18:
  v19 = *(_QWORD *)v8 == 0;
LABEL_19:
  if ( v19 || !v13 )
  {
    result = 0;
    goto LABEL_22;
  }
  if ( v16 <= 0x40 )
  {
    if ( v17 != (_QWORD *)2 )
    {
      if ( v17 == (_QWORD *)1 && v10 == 37 )
        goto LABEL_30;
      return 0;
    }
    if ( v10 != 36 || v32 != 34 )
      return 0;
LABEL_31:
    result = sub_AD6450(v27);
LABEL_22:
    if ( v16 <= 0x40 )
      return result;
    goto LABEL_23;
  }
  if ( v16 - (unsigned int)sub_C444A0((__int64)&v34) <= 0x40 )
  {
    v26 = *v17;
    if ( *v17 != 2 )
      goto LABEL_70;
    if ( v10 != 36 )
      goto LABEL_69;
    if ( v32 == 34 )
      goto LABEL_31;
    if ( v16 - (unsigned int)sub_C444A0((__int64)&v34) <= 0x40 )
    {
LABEL_69:
      v26 = *v17;
LABEL_70:
      if ( v26 != 1 )
      {
        result = 0;
        goto LABEL_24;
      }
      if ( v10 == 37 )
      {
LABEL_30:
        result = 0;
        if ( v32 != 34 )
          goto LABEL_22;
        goto LABEL_31;
      }
    }
  }
  result = 0;
LABEL_23:
  if ( v17 )
  {
LABEL_24:
    v33 = result;
    j_j___libc_free_0_0(v17);
    return v33;
  }
  return result;
}
