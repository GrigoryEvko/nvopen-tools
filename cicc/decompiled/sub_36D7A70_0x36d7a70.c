// Function: sub_36D7A70
// Address: 0x36d7a70
//
__int64 __fastcall sub_36D7A70(__int64 a1, __int64 a2)
{
  unsigned __int64 v3; // rdi
  char v4; // r13
  __int64 v5; // rax
  __int64 *v6; // rbx
  __int64 v7; // r15
  __int64 *v8; // r14
  __int64 v9; // rax
  __int64 v10; // r15
  __int64 *v11; // r15
  char *v12; // r12
  char v13; // al
  __int64 *v15; // rdx
  char v16; // al
  char v17; // al
  char v18; // al
  char v19; // al
  char v20; // al
  char v21; // al
  char v22; // al
  char v23; // al
  char v24; // al
  __int64 *v25; // [rsp+8h] [rbp-88h]
  __int64 *v26; // [rsp+10h] [rbp-80h] BYREF
  __int64 v27; // [rsp+18h] [rbp-78h]
  _BYTE v28[112]; // [rsp+20h] [rbp-70h] BYREF

  v3 = 0;
  v4 = sub_CE9220(a2);
  v26 = (__int64 *)v28;
  v27 = 0x800000000LL;
  v5 = **(_QWORD **)(a1 + 112);
  if ( v5 && (v5 & 4) == 0 )
    v3 = v5 & 0xFFFFFFFFFFFFFFF8LL;
  sub_98B4D0(v3, (__int64)&v26, 0, 6u);
  v6 = v26;
  v7 = 8LL * (unsigned int)v27;
  v8 = &v26[(unsigned __int64)v7 / 8];
  v9 = v7 >> 3;
  v10 = v7 >> 5;
  if ( v10 )
  {
    v11 = &v26[4 * v10];
    while ( 1 )
    {
      while ( 1 )
      {
        v12 = (char *)*v6;
        v13 = *(_BYTE *)*v6;
        if ( v13 != 22 )
          break;
        if ( !v4 || !(unsigned __int8)sub_B2BD80(*v6) || !(unsigned __int8)sub_B2D700((__int64)v12) )
          goto LABEL_8;
        v12 = (char *)v6[1];
        v15 = v6 + 1;
        v16 = *v12;
        if ( *v12 != 22 )
          goto LABEL_20;
LABEL_39:
        v25 = v15;
        v22 = sub_B2BD80((__int64)v12);
        v15 = v25;
        if ( !v22 )
          goto LABEL_16;
        if ( !(unsigned __int8)sub_B2D700((__int64)v12) )
        {
LABEL_41:
          LOBYTE(v12) = v8 == v25;
          goto LABEL_9;
        }
        v12 = (char *)v6[2];
        v15 = v6 + 2;
        v17 = *v12;
        if ( *v12 != 22 )
          goto LABEL_43;
LABEL_24:
        v25 = v15;
        v18 = sub_B2BD80((__int64)v12);
        v15 = v25;
        if ( !v18 )
          goto LABEL_16;
        if ( !(unsigned __int8)sub_B2D700((__int64)v12) )
          goto LABEL_41;
        v12 = (char *)v6[3];
        v15 = v6 + 3;
        v19 = *v12;
        if ( *v12 != 22 )
          goto LABEL_48;
LABEL_27:
        v25 = v15;
        v20 = sub_B2BD80((__int64)v12);
        v15 = v25;
        if ( !v20 )
          goto LABEL_16;
        if ( !(unsigned __int8)sub_B2D700((__int64)v12) )
          goto LABEL_41;
        v6 += 4;
        if ( v6 == v11 )
          goto LABEL_30;
      }
      if ( v13 != 3 || (v12[80] & 1) == 0 )
        goto LABEL_8;
      v12 = (char *)v6[1];
      v15 = v6 + 1;
      v16 = *v12;
      if ( *v12 == 22 )
      {
        if ( !v4 )
          goto LABEL_16;
        goto LABEL_39;
      }
LABEL_20:
      if ( v16 != 3 || (v12[80] & 1) == 0 )
      {
LABEL_16:
        LOBYTE(v12) = v8 == v15;
        goto LABEL_9;
      }
      v12 = (char *)v6[2];
      v15 = v6 + 2;
      v17 = *v12;
      if ( *v12 == 22 )
      {
        if ( !v4 )
          goto LABEL_16;
        goto LABEL_24;
      }
LABEL_43:
      if ( v17 != 3 || (v12[80] & 1) == 0 )
        goto LABEL_16;
      v12 = (char *)v6[3];
      v15 = v6 + 3;
      v19 = *v12;
      if ( *v12 == 22 )
      {
        if ( !v4 )
          goto LABEL_16;
        goto LABEL_27;
      }
LABEL_48:
      if ( v19 != 3 || (v12[80] & 1) == 0 )
        goto LABEL_16;
      v6 += 4;
      if ( v6 == v11 )
      {
LABEL_30:
        v9 = v8 - v6;
        break;
      }
    }
  }
  if ( v9 != 2 )
  {
    if ( v9 != 3 )
    {
      LODWORD(v12) = 1;
      if ( v9 != 1 )
        goto LABEL_9;
      goto LABEL_34;
    }
    v12 = (char *)*v6;
    v23 = *(_BYTE *)*v6;
    if ( v23 == 22 )
    {
      if ( !v4 || !(unsigned __int8)sub_B2BD80(*v6) || !(unsigned __int8)sub_B2D700((__int64)v12) )
        goto LABEL_8;
    }
    else if ( v23 != 3 || (v12[80] & 1) == 0 )
    {
      goto LABEL_8;
    }
    ++v6;
  }
  v12 = (char *)*v6;
  v24 = *(_BYTE *)*v6;
  if ( v24 == 22 )
  {
    if ( !v4 || !(unsigned __int8)sub_B2BD80(*v6) || !(unsigned __int8)sub_B2D700((__int64)v12) )
      goto LABEL_8;
  }
  else if ( v24 != 3 || (v12[80] & 1) == 0 )
  {
    goto LABEL_8;
  }
  ++v6;
LABEL_34:
  v12 = (char *)*v6;
  v21 = *(_BYTE *)*v6;
  if ( v21 == 22 )
  {
    if ( !v4 || !(unsigned __int8)sub_B2BD80(*v6) || (LODWORD(v12) = sub_B2D700((__int64)v12), !(_BYTE)v12) )
LABEL_8:
      LOBYTE(v12) = v8 == v6;
  }
  else
  {
    if ( v21 != 3 )
      goto LABEL_8;
    LODWORD(v12) = v12[80] & 1;
    if ( !(_DWORD)v12 )
      goto LABEL_8;
  }
LABEL_9:
  if ( v26 != (__int64 *)v28 )
    _libc_free((unsigned __int64)v26);
  return (unsigned int)v12;
}
