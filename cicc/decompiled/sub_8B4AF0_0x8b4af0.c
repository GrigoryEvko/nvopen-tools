// Function: sub_8B4AF0
// Address: 0x8b4af0
//
__int64 __fastcall sub_8B4AF0(__int64 *a1, __int64 *a2, __int64 *a3, __int64 a4, unsigned int a5)
{
  __int64 *v5; // r15
  _BOOL4 v6; // r14d
  unsigned int v8; // r12d
  int v9; // ebx
  __int64 v10; // rdi
  __int64 *v11; // rax
  __int64 **v12; // rdx
  char v13; // cl
  unsigned int v14; // r8d
  unsigned int v15; // eax
  __int64 v16; // rax
  __int64 v17; // rdi
  __int64 *v18; // rax
  __int64 v19; // rdx
  bool v20; // dl
  __int64 v22; // rdi
  unsigned int v23; // eax
  unsigned int v24; // eax
  __int64 *v27; // [rsp+20h] [rbp-50h] BYREF
  __int64 *v28; // [rsp+28h] [rbp-48h] BYREF
  __int64 v29[7]; // [rsp+38h] [rbp-38h] BYREF

  v5 = 0;
  v6 = 0;
  v8 = 1;
  v9 = 0;
  v28 = a1;
  v27 = a2;
  v29[0] = 0;
  while ( 1 )
  {
    if ( !v28 )
      goto LABEL_41;
    v6 = *((_BYTE *)v28 + 8) == 3;
    if ( *((_BYTE *)v28 + 8) == 3 )
    {
      sub_72F220(&v28);
      if ( !v28 )
      {
        v6 = 1;
        goto LABEL_41;
      }
    }
    v11 = v27;
    if ( !v27 )
      goto LABEL_41;
    if ( *((_BYTE *)v27 + 8) != 3 )
    {
      v9 = 0;
      v5 = v27;
      if ( !v29[0] )
        goto LABEL_37;
      goto LABEL_11;
    }
    sub_72F220(&v27);
    v11 = v27;
    if ( !v27 )
      break;
    v5 = v27;
    v9 = 1;
    if ( !v29[0] )
    {
LABEL_37:
      v22 = v11[2];
      if ( v22 )
      {
        sub_869480(v22, a4, a3, v29);
        v11 = v27;
      }
    }
LABEL_11:
    v12 = (__int64 **)v28;
    v13 = *((_BYTE *)v28 + 8);
    if ( (v28[3] & 0x10) != 0 )
    {
      v8 = 0;
      if ( *((_BYTE *)v11 + 8) != v13 || (v11[3] & 0x10) == 0 )
        goto LABEL_3;
      v14 = a5 | 0x800;
      if ( !v13 )
      {
LABEL_47:
        v23 = sub_8B3500((__m128i *)v28[4], v11[4], a3, a4, v14 & 0xA00 | 0x400);
        v12 = (__int64 **)v28;
        v8 = v23;
        goto LABEL_3;
      }
LABEL_16:
      if ( v13 == 1 )
      {
        v24 = sub_8B46F0(v28[4], v11[4], a3, a4, v14);
        v12 = (__int64 **)v28;
        v8 = v24;
        goto LABEL_3;
      }
      if ( v13 != 2 )
        sub_721090();
      v15 = sub_8B30E0(*(_QWORD *)v28[4], *(_QWORD *)v11[4], a3, a4);
      v10 = v29[0];
      v8 = v15;
      v28 = (__int64 *)*v28;
      if ( v29[0] )
        goto LABEL_4;
LABEL_19:
      v16 = *v27;
      v27 = (__int64 *)v16;
      if ( !v16 )
        goto LABEL_5;
      v9 = 0;
      if ( *(_BYTE *)(v16 + 8) != 3 )
        goto LABEL_5;
      v9 = 1;
      sub_72F220(&v27);
      if ( !v8 )
      {
LABEL_22:
        v17 = v29[0];
        if ( !v29[0] )
          goto LABEL_24;
        goto LABEL_23;
      }
    }
    else
    {
      v14 = a5 & 0xFFFFF7FF;
      if ( (v28[3] & 8) == 0 )
        v14 = a5;
      v8 = 0;
      if ( v13 == *((_BYTE *)v11 + 8) )
      {
        if ( !v13 )
          goto LABEL_47;
        goto LABEL_16;
      }
LABEL_3:
      v10 = v29[0];
      v28 = *v12;
      if ( !v29[0] )
        goto LABEL_19;
LABEL_4:
      sub_866B90(v10);
LABEL_5:
      if ( !v8 )
        goto LABEL_22;
    }
  }
  v9 = 1;
LABEL_41:
  v17 = v29[0];
  if ( !v29[0] )
  {
    if ( !v5 )
      goto LABEL_43;
LABEL_24:
    v18 = v28;
    if ( !v28 || (v19 = *v5, v27) || !v19 )
    {
      if ( v6 )
        return v8;
      goto LABEL_28;
    }
    if ( *(_BYTE *)(v19 + 8) == 3 || v6 )
      return v8;
    return v18 && v9 && (a5 & 0x80) != 0;
  }
LABEL_23:
  sub_866BE0(v17);
  v27 = 0;
  if ( v5 )
    goto LABEL_24;
LABEL_43:
  v18 = v28;
  if ( v6 )
    return v8;
LABEL_28:
  v20 = 1;
  if ( v27 )
    v20 = v27[2] != 0;
  if ( (v18 == 0) != v20 )
    return v18 && v9 && (a5 & 0x80) != 0;
  return v8;
}
