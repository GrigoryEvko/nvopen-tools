// Function: sub_852210
// Address: 0x852210
//
__int64 __fastcall sub_852210(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  size_t v6; // rsi
  char *v7; // rdi
  __int64 v8; // r13
  char *v10; // rax
  unsigned __int8 *v11; // r12
  unsigned __int8 *v12; // rax
  __int64 v13; // rbx
  char *v14; // rax
  _QWORD *v15; // r12
  __int64 v16; // r12
  __int64 v17; // rax
  __int64 v18; // r14
  char *v19; // r12
  int v20; // [rsp+8h] [rbp-88h]
  int v21; // [rsp+Ch] [rbp-84h]
  __int64 v22; // [rsp+18h] [rbp-78h] BYREF
  __time_t ptr; // [rsp+20h] [rbp-70h] BYREF
  int v24; // [rsp+28h] [rbp-68h]
  int v25; // [rsp+2Ch] [rbp-64h]

  v6 = size;
  dword_4F5F928 = 0;
  LODWORD(ptr) = 0;
  v7 = ::ptr;
  if ( qword_4F5F850 < size )
  {
    v13 = qword_4F5F850 + 1024;
    if ( size >= qword_4F5F850 + 1024 )
      v13 = size;
    v14 = (char *)sub_822C60(::ptr, qword_4F5F850, v13, a4, qword_4F5F850, a6);
    qword_4F5F850 = v13;
    v6 = size;
    ::ptr = v14;
    v7 = v14;
  }
  if ( fread(v7, v6, 1u, qword_4F5FB48) != 1 || (v21 = 1, strncmp(::ptr, byte_4F5FB80, size)) )
  {
    dword_4F5F928 = 626;
    v21 = 0;
  }
  if ( fread(&ptr, 4u, 1u, qword_4F5FB48) != 1 || !(_DWORD)ptr )
  {
    dword_4F5F928 = 2226;
    return 0;
  }
  if ( !v21 )
    return 0;
  v10 = sub_852050();
  if ( strcmp(v10, qword_4F076B0) )
  {
    dword_4F5F928 = 627;
    sub_852050();
    return 0;
  }
  v11 = (unsigned __int8 *)sub_852050();
  v12 = (unsigned __int8 *)sub_722430(qword_4F076F0, 0);
  if ( sub_722B80(v11, v12, 0) )
  {
    dword_4F5F928 = 629;
    return 0;
  }
  v15 = (_QWORD *)qword_4F5FB60;
  if ( qword_4F5FB60 )
  {
    while ( (unsigned int)sub_852110(&ptr) && (unsigned int)sub_851F00((__int64)v15, (__int64)&ptr) )
    {
      v15 = (_QWORD *)*v15;
      if ( !v15 )
        goto LABEL_25;
    }
    goto LABEL_24;
  }
LABEL_25:
  v20 = sub_852110(&ptr);
  if ( v20 )
  {
LABEL_24:
    dword_4F5F928 = 629;
    return 0;
  }
  v16 = qword_4F5FB70;
  if ( qword_4F5FB70 )
  {
    v17 = qword_4F5FB70;
    do
    {
      *(_BYTE *)(v17 + 48) = 0;
      v17 = *(_QWORD *)v17;
    }
    while ( v17 );
  }
  v8 = 0;
  while ( (unsigned int)sub_852110(&ptr) )
  {
    if ( !v16 )
    {
LABEL_62:
      dword_4F5F928 = 630;
      return 0;
    }
    if ( v24 != 2 )
      sub_721090();
    v18 = v16;
    if ( v25 == 9 )
    {
      while ( 1 )
      {
        if ( *(_QWORD *)(v18 + 8) != 0x900000002LL )
          goto LABEL_41;
        if ( !*(_BYTE *)(v18 + 48) && (unsigned int)sub_851F00(v18, (__int64)&ptr) )
          break;
        v18 = *(_QWORD *)v18;
        if ( !v18 )
          goto LABEL_41;
      }
      *(_BYTE *)(v18 + 48) = 1;
    }
    else
    {
      while ( *(_QWORD *)(v18 + 8) == 0x900000002LL && *(_BYTE *)(v18 + 48) )
      {
        v18 = *(_QWORD *)v18;
        if ( !v18 )
          goto LABEL_41;
      }
      if ( !(unsigned int)sub_851F00(v18, (__int64)&ptr) )
        goto LABEL_41;
      v16 = *(_QWORD *)v18;
      v8 = v18;
    }
  }
  if ( !v16 )
  {
LABEL_43:
    if ( v8 )
      goto LABEL_46;
    return 0;
  }
  v20 = v21;
  while ( 1 )
  {
LABEL_41:
    if ( *(_QWORD *)(v16 + 8) != 0x900000002LL || !*(_BYTE *)(v16 + 48) )
    {
      if ( !v20 )
        goto LABEL_62;
      goto LABEL_43;
    }
    v8 = v16;
    if ( !*(_QWORD *)v16 )
      break;
    v16 = *(_QWORD *)v16;
  }
  if ( !v20 )
    goto LABEL_62;
LABEL_46:
  while ( 1 )
  {
    v19 = sub_852050();
    if ( !*v19 )
      break;
    if ( fread(&v22, 8u, 1u, qword_4F5FB48) != 1 )
      sub_851ED0();
    if ( !(unsigned int)sub_723E40((__int64)v19, &ptr) || v22 != ptr )
    {
      dword_4F5F928 = 628;
      if ( !unk_4D044E8 )
        return 0;
      if ( qword_4F5FB48 )
        fclose(qword_4F5FB48);
      v8 = 0;
      qword_4F5FB48 = 0;
      sub_7212E0(unk_4D044F8);
      return v8;
    }
  }
  return v8;
}
