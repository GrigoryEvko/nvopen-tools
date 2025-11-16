// Function: sub_21BD8E0
// Address: 0x21bd8e0
//
__int64 __fastcall sub_21BD8E0(__int64 a1, __int64 *a2)
{
  char v3; // r13
  unsigned __int64 v4; // rax
  unsigned __int64 v5; // rdi
  __int64 *v6; // rbx
  __int64 v7; // r15
  __int64 *v8; // r14
  __int64 v9; // rax
  __int64 v10; // r15
  __int64 *v11; // r15
  __int64 v12; // r12
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

  v3 = sub_1C2F070(*a2);
  v26 = (__int64 *)v28;
  v27 = 0x800000000LL;
  v4 = sub_1E0A0C0((__int64)a2);
  v5 = **(_QWORD **)(a1 + 104) & 0xFFFFFFFFFFFFFFF8LL;
  if ( (**(_QWORD **)(a1 + 104) & 4) != 0 )
    v5 = 0;
  sub_14AD470(v5, (__int64)&v26, v4, 0, 6u);
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
        v12 = *v6;
        v13 = *(_BYTE *)(*v6 + 16);
        if ( v13 != 17 )
          break;
        if ( !v3 || !(unsigned __int8)sub_15E03C0(*v6) || !(unsigned __int8)sub_15E04B0(v12) )
          goto LABEL_7;
        v12 = v6[1];
        v15 = v6 + 1;
        v16 = *(_BYTE *)(v12 + 16);
        if ( v16 != 17 )
          goto LABEL_19;
LABEL_38:
        v25 = v15;
        v22 = sub_15E03C0(v12);
        v15 = v25;
        if ( !v22 )
          goto LABEL_15;
        if ( !(unsigned __int8)sub_15E04B0(v12) )
        {
LABEL_40:
          LOBYTE(v12) = v8 == v25;
          goto LABEL_8;
        }
        v12 = v6[2];
        v15 = v6 + 2;
        v17 = *(_BYTE *)(v12 + 16);
        if ( v17 != 17 )
          goto LABEL_42;
LABEL_23:
        v25 = v15;
        v18 = sub_15E03C0(v12);
        v15 = v25;
        if ( !v18 )
          goto LABEL_15;
        if ( !(unsigned __int8)sub_15E04B0(v12) )
          goto LABEL_40;
        v12 = v6[3];
        v15 = v6 + 3;
        v19 = *(_BYTE *)(v12 + 16);
        if ( v19 != 17 )
          goto LABEL_47;
LABEL_26:
        v25 = v15;
        v20 = sub_15E03C0(v12);
        v15 = v25;
        if ( !v20 )
          goto LABEL_15;
        if ( !(unsigned __int8)sub_15E04B0(v12) )
          goto LABEL_40;
        v6 += 4;
        if ( v6 == v11 )
          goto LABEL_29;
      }
      if ( v13 != 3 || (*(_BYTE *)(v12 + 80) & 1) == 0 )
        goto LABEL_7;
      v12 = v6[1];
      v15 = v6 + 1;
      v16 = *(_BYTE *)(v12 + 16);
      if ( v16 == 17 )
      {
        if ( !v3 )
          goto LABEL_15;
        goto LABEL_38;
      }
LABEL_19:
      if ( v16 != 3 || (*(_BYTE *)(v12 + 80) & 1) == 0 )
      {
LABEL_15:
        LOBYTE(v12) = v8 == v15;
        goto LABEL_8;
      }
      v12 = v6[2];
      v15 = v6 + 2;
      v17 = *(_BYTE *)(v12 + 16);
      if ( v17 == 17 )
      {
        if ( !v3 )
          goto LABEL_15;
        goto LABEL_23;
      }
LABEL_42:
      if ( v17 != 3 || (*(_BYTE *)(v12 + 80) & 1) == 0 )
        goto LABEL_15;
      v12 = v6[3];
      v15 = v6 + 3;
      v19 = *(_BYTE *)(v12 + 16);
      if ( v19 == 17 )
      {
        if ( !v3 )
          goto LABEL_15;
        goto LABEL_26;
      }
LABEL_47:
      if ( v19 != 3 || (*(_BYTE *)(v12 + 80) & 1) == 0 )
        goto LABEL_15;
      v6 += 4;
      if ( v6 == v11 )
      {
LABEL_29:
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
        goto LABEL_8;
      goto LABEL_33;
    }
    v12 = *v6;
    v23 = *(_BYTE *)(*v6 + 16);
    if ( v23 == 17 )
    {
      if ( !v3 || !(unsigned __int8)sub_15E03C0(*v6) || !(unsigned __int8)sub_15E04B0(v12) )
        goto LABEL_7;
    }
    else if ( v23 != 3 || (*(_BYTE *)(v12 + 80) & 1) == 0 )
    {
      goto LABEL_7;
    }
    ++v6;
  }
  v12 = *v6;
  v24 = *(_BYTE *)(*v6 + 16);
  if ( v24 == 17 )
  {
    if ( !v3 || !(unsigned __int8)sub_15E03C0(*v6) || !(unsigned __int8)sub_15E04B0(v12) )
      goto LABEL_7;
  }
  else if ( v24 != 3 || (*(_BYTE *)(v12 + 80) & 1) == 0 )
  {
    goto LABEL_7;
  }
  ++v6;
LABEL_33:
  v12 = *v6;
  v21 = *(_BYTE *)(*v6 + 16);
  if ( v21 == 17 )
  {
    if ( !v3 || !(unsigned __int8)sub_15E03C0(*v6) || (LODWORD(v12) = sub_15E04B0(v12), !(_BYTE)v12) )
LABEL_7:
      LOBYTE(v12) = v8 == v6;
  }
  else
  {
    if ( v21 != 3 )
      goto LABEL_7;
    LODWORD(v12) = *(_BYTE *)(v12 + 80) & 1;
    if ( !(_DWORD)v12 )
      goto LABEL_7;
  }
LABEL_8:
  if ( v26 != (__int64 *)v28 )
    _libc_free((unsigned __int64)v26);
  return (unsigned int)v12;
}
