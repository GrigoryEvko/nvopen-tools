// Function: sub_3156450
// Address: 0x3156450
//
__int64 *__fastcall sub_3156450(__int64 *a1, __int64 a2, _QWORD *a3)
{
  __int64 v5; // rdx
  __int64 v6; // rcx
  char v7; // r13
  __int64 v8; // r8
  __int64 v9; // r9
  int v10; // edx
  _QWORD *v11; // r15
  unsigned __int64 v12; // rcx
  _QWORD *v13; // rax
  char v14; // di
  _BOOL4 v15; // r9d
  _QWORD *v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r9
  __int64 v20; // r8
  __int64 v21; // rsi
  char v22; // al
  bool v24; // [rsp+Ch] [rbp-174h]
  _BOOL4 v26; // [rsp+18h] [rbp-168h]
  _QWORD *v27; // [rsp+18h] [rbp-168h]
  _QWORD *v28; // [rsp+20h] [rbp-160h]
  const char *v29; // [rsp+30h] [rbp-150h] BYREF
  char *v30; // [rsp+38h] [rbp-148h] BYREF
  __int64 v31; // [rsp+40h] [rbp-140h]
  _BYTE v32[24]; // [rsp+48h] [rbp-138h] BYREF
  __int64 v33; // [rsp+60h] [rbp-120h] BYREF
  __int64 v34; // [rsp+68h] [rbp-118h]
  __int64 *v35; // [rsp+70h] [rbp-110h]
  const char *v36; // [rsp+78h] [rbp-108h]
  char *v37; // [rsp+80h] [rbp-100h] BYREF
  unsigned int v38; // [rsp+88h] [rbp-F8h]
  _BYTE v39[144]; // [rsp+90h] [rbp-F0h] BYREF
  _QWORD *v40; // [rsp+120h] [rbp-60h]
  char v41; // [rsp+140h] [rbp-40h]

  sub_3154320(&v33, a2, 12);
  if ( (v33 & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    *a1 = v33 & 0xFFFFFFFFFFFFFFFELL | 1;
    return a1;
  }
  v28 = a3 + 1;
  while ( 1 )
  {
    v7 = sub_3154C60(a2, 13, v5, v6);
    if ( !v7 )
    {
      *a1 = 1;
      return a1;
    }
    sub_31552E0((__int64)&v33, a2, 13);
    v10 = v41 & 1;
    v41 = (2 * v10) | v41 & 0xFD;
    if ( (_BYTE)v10 )
    {
      *a1 = v33 | 1;
      return a1;
    }
    v29 = v36;
    v30 = v32;
    v31 = 0x100000000LL;
    if ( v38 )
      break;
    v11 = (_QWORD *)a3[2];
    if ( !v11 )
      goto LABEL_42;
    while ( 1 )
    {
LABEL_9:
      v12 = v11[4];
      v13 = (_QWORD *)v11[3];
      v14 = 0;
      if ( (unsigned __int64)v29 < v12 )
      {
        v13 = (_QWORD *)v11[2];
        v14 = v7;
      }
      if ( !v13 )
        break;
      v11 = v13;
    }
    if ( !v14 )
    {
      if ( (unsigned __int64)v29 <= v12 )
        goto LABEL_29;
      goto LABEL_14;
    }
    if ( v11 != (_QWORD *)a3[3] )
      goto LABEL_28;
LABEL_14:
    v15 = 1;
    if ( v11 != v28 )
      v15 = (unsigned __int64)v29 < v11[4];
LABEL_16:
    v26 = v15;
    v16 = (_QWORD *)sub_22077B0(0x40u);
    v19 = v26;
    v20 = (__int64)v16;
    v16[4] = v29;
    v16[5] = v16 + 7;
    v16[6] = 0x100000000LL;
    if ( (_DWORD)v31 )
    {
      v24 = v26;
      v27 = v16;
      sub_3153680((__int64)(v16 + 5), &v30, v17, v18, (__int64)v16, v19);
      LOBYTE(v19) = v24;
      v20 = (__int64)v27;
    }
    v21 = v20;
    sub_220F040(v19, v20, v11, v28);
    ++a3[5];
    if ( v30 == v32 )
    {
      v22 = v41;
      if ( (v41 & 2) != 0 )
        goto LABEL_47;
    }
    else
    {
      _libc_free((unsigned __int64)v30);
      v22 = v41;
      if ( (v41 & 2) != 0 )
        goto LABEL_47;
    }
    if ( (v22 & 1) != 0 )
    {
      if ( v33 )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v33 + 8LL))(v33);
    }
    else
    {
      sub_31541A0(v40);
      if ( v37 != v39 )
        _libc_free((unsigned __int64)v37);
      if ( v35 )
      {
        v5 = v34;
        *v35 = v34;
      }
      if ( v34 )
      {
        v5 = (__int64)v35;
        *(_QWORD *)(v34 + 8) = v35;
      }
    }
  }
  sub_3153680((__int64)&v30, &v37, v38, (unsigned int)(2 * v10), v8, v9);
  v11 = (_QWORD *)a3[2];
  if ( v11 )
    goto LABEL_9;
LABEL_42:
  v11 = a3 + 1;
  if ( v28 == (_QWORD *)a3[3] )
  {
    v11 = a3 + 1;
    v15 = 1;
    goto LABEL_16;
  }
LABEL_28:
  if ( *(_QWORD *)(sub_220EF80((__int64)v11) + 32) < (unsigned __int64)v29 )
    goto LABEL_14;
LABEL_29:
  if ( v30 != v32 )
    _libc_free((unsigned __int64)v30);
  v21 = a2;
  v29 = "Duplicate flat profile entries";
  v32[9] = 1;
  v32[8] = 3;
  sub_31542E0(a1, a2, (void **)&v29);
  if ( (v41 & 2) != 0 )
LABEL_47:
    sub_31551A0(&v33, v21);
  if ( (v41 & 1) != 0 )
  {
    if ( v33 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v33 + 8LL))(v33);
  }
  else
  {
    sub_31541A0(v40);
    if ( v37 != v39 )
      _libc_free((unsigned __int64)v37);
    if ( v35 )
      *v35 = v34;
    if ( v34 )
      *(_QWORD *)(v34 + 8) = v35;
  }
  return a1;
}
