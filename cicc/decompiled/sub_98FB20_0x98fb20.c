// Function: sub_98FB20
// Address: 0x98fb20
//
char __fastcall sub_98FB20(_BYTE *a1, _BYTE *a2)
{
  char result; // al
  __int64 v3; // r14
  __int64 v4; // r13
  __int64 v5; // rbx
  __int64 v6; // rdx
  __int64 v7; // r8
  unsigned __int64 v8; // rax
  unsigned int v9; // r12d
  unsigned __int64 v10; // rax
  __int64 v11; // r8
  _BYTE *v12; // r13
  _BYTE *v13; // r14
  unsigned int v14; // edi
  __int64 v15; // rdx
  __int64 v16; // rsi
  unsigned int v17; // edi
  bool v18; // zf
  __int64 v19; // rdx
  unsigned int v20; // ebx
  _BYTE *v21; // rax
  unsigned __int64 v22; // rax
  _BYTE *v23; // rax
  __int64 v24; // [rsp-A0h] [rbp-A0h]
  char v25; // [rsp-A0h] [rbp-A0h]
  char v26; // [rsp-A0h] [rbp-A0h]
  char v27; // [rsp-A0h] [rbp-A0h]
  char v28; // [rsp-A0h] [rbp-A0h]
  char v29; // [rsp-A0h] [rbp-A0h]
  __int64 v30; // [rsp-A0h] [rbp-A0h]
  __int64 v31; // [rsp-A0h] [rbp-A0h]
  char v32; // [rsp-A0h] [rbp-A0h]
  __int64 v33; // [rsp-98h] [rbp-98h] BYREF
  unsigned int v34; // [rsp-90h] [rbp-90h]
  __int64 v35; // [rsp-88h] [rbp-88h]
  unsigned int v36; // [rsp-80h] [rbp-80h]
  __int64 v37; // [rsp-78h] [rbp-78h] BYREF
  unsigned int v38; // [rsp-70h] [rbp-70h]
  __int64 v39; // [rsp-68h] [rbp-68h] BYREF
  unsigned int v40; // [rsp-60h] [rbp-60h]
  __int64 v41; // [rsp-58h] [rbp-58h] BYREF
  unsigned int v42; // [rsp-50h] [rbp-50h]
  __int64 v43; // [rsp-48h] [rbp-48h] BYREF
  unsigned int v44; // [rsp-40h] [rbp-40h]

  if ( *a1 != 82 )
    return 0;
  v3 = *((_QWORD *)a1 - 8);
  if ( !v3 )
    return 0;
  v4 = *((_QWORD *)a1 - 4);
  if ( !v4 )
    return 0;
  v5 = sub_B53900(a1);
  if ( *a2 != 82 )
    return 0;
  v6 = *((_QWORD *)a2 - 8);
  v7 = *((_QWORD *)a2 - 4);
  if ( v3 == v6 )
  {
    v31 = *((_QWORD *)a2 - 4);
    if ( !v7 )
      return 0;
    v22 = sub_B53900(a2);
    v11 = v31;
    v9 = v22;
    v10 = HIDWORD(v22);
  }
  else
  {
    v24 = *((_QWORD *)a2 - 8);
    if ( !v6 || v3 != v7 )
      return 0;
    v8 = sub_B53960(a2);
    v9 = v8;
    v10 = HIDWORD(v8);
    v11 = v24;
  }
  if ( BYTE4(v5) != (_BYTE)v10 )
    return 0;
  if ( v4 == v11 )
    return (unsigned int)sub_B52870(v9) == (_DWORD)v5;
  if ( *(_BYTE *)v4 == 17 )
  {
    v12 = (_BYTE *)(v4 + 24);
  }
  else
  {
    v30 = v11;
    if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v4 + 8) + 8LL) - 17 > 1 )
      return 0;
    if ( *(_BYTE *)v4 > 0x15u )
      return 0;
    v21 = (_BYTE *)sub_AD7630(v4, 0);
    if ( !v21 || *v21 != 17 )
      return 0;
    v11 = v30;
    v12 = v21 + 24;
  }
  v13 = (_BYTE *)(v11 + 24);
  if ( *(_BYTE *)v11 != 17 )
  {
    if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v11 + 8) + 8LL) - 17 <= 1 && *(_BYTE *)v11 <= 0x15u )
    {
      v23 = (_BYTE *)sub_AD7630(v11, 0);
      if ( v23 )
      {
        if ( *v23 == 17 )
        {
          v13 = v23 + 24;
          goto LABEL_17;
        }
      }
    }
    return 0;
  }
LABEL_17:
  if ( BYTE4(v5) )
  {
    v14 = *((_DWORD *)v12 + 2);
    v15 = *(_QWORD *)v12;
    v16 = 1LL << ((unsigned __int8)v14 - 1);
    if ( v14 > 0x40 )
      v15 = *(_QWORD *)(v15 + 8LL * ((v14 - 1) >> 6));
    v17 = *((_DWORD *)v13 + 2);
    v18 = (v15 & v16) == 0;
    v19 = *(_QWORD *)v13;
    if ( v17 > 0x40 )
      v19 = *(_QWORD *)(v19 + 8LL * ((v17 - 1) >> 6));
    if ( ((v19 & (1LL << ((unsigned __int8)v17 - 1))) == 0) != v18 )
      return 0;
  }
  sub_AB1A50(&v33, (unsigned int)v5, v12);
  sub_AB1A50(&v37, v9, v13);
  sub_ABB300(&v41, &v33);
  v20 = v42;
  if ( v42 <= 0x40 )
  {
    if ( v41 != v37 )
    {
      result = 0;
      if ( v44 <= 0x40 )
        goto LABEL_28;
      goto LABEL_55;
    }
  }
  else
  {
    result = sub_C43C50(&v41, &v37);
    if ( !result )
    {
      if ( v44 <= 0x40 )
        goto LABEL_26;
      goto LABEL_55;
    }
  }
  if ( v44 <= 0x40 )
  {
    result = v43 == v39;
    goto LABEL_57;
  }
  result = sub_C43C50(&v43, &v39);
LABEL_55:
  if ( v43 )
  {
    v32 = result;
    j_j___libc_free_0_0(v43);
    v20 = v42;
    result = v32;
  }
LABEL_57:
  if ( v20 > 0x40 )
  {
LABEL_26:
    if ( v41 )
    {
      v25 = result;
      j_j___libc_free_0_0(v41);
      result = v25;
    }
  }
LABEL_28:
  if ( v40 > 0x40 && v39 )
  {
    v26 = result;
    j_j___libc_free_0_0(v39);
    result = v26;
  }
  if ( v38 > 0x40 && v37 )
  {
    v27 = result;
    j_j___libc_free_0_0(v37);
    result = v27;
  }
  if ( v36 > 0x40 && v35 )
  {
    v28 = result;
    j_j___libc_free_0_0(v35);
    result = v28;
  }
  if ( v34 > 0x40 )
  {
    if ( v33 )
    {
      v29 = result;
      j_j___libc_free_0_0(v33);
      return v29;
    }
  }
  return result;
}
