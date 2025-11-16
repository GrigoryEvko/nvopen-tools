// Function: sub_2C46F30
// Address: 0x2c46f30
//
__int64 __fastcall sub_2C46F30(__int64 a1)
{
  __int64 v2; // r13
  unsigned int v3; // eax
  unsigned int v4; // r14d
  _QWORD *v5; // rax
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rax
  _QWORD *v9; // rdi
  __int64 (__fastcall *v10)(__int64); // rax
  __int64 v11; // rax
  char v12; // al
  char *v13; // rbx
  __int64 v14; // r12
  char *v15; // r13
  __int64 v16; // r12
  char *v17; // r12
  __int64 v19; // rax
  __int64 v20; // r12
  __int64 v21; // rax
  __int64 v22; // r12
  char *v23; // r12
  signed __int64 v24; // rax

  while ( 1 )
  {
    if ( !sub_2BF04A0(a1) )
      return 1;
    v2 = sub_2BF0490(a1);
    if ( v2 )
    {
      LOBYTE(v3) = sub_2BFB0D0(a1);
      v4 = v3;
      if ( (_BYTE)v3 )
        break;
    }
    v5 = (_QWORD *)sub_2BF9BD0(*(_QWORD *)(v2 + 80));
    v6 = sub_2BF3F10(v5);
    v7 = sub_2BF04D0(v6);
    if ( v7 + 112 == (*(_QWORD *)(v7 + 112) & 0xFFFFFFFFFFFFFFF8LL) )
    {
      if ( *(_DWORD *)(v7 + 88) != 1 )
        BUG();
      v7 = **(_QWORD **)(v7 + 80);
    }
    v8 = *(_QWORD *)(v7 + 120);
    v9 = 0;
    if ( v8 )
    {
      v9 = (_QWORD *)(v8 - 24);
      v8 += 72;
    }
    if ( a1 == v8 )
      return 1;
    v10 = *(__int64 (__fastcall **)(__int64))(*v9 + 40LL);
    v11 = v10 == sub_2AA7530 ? *(_QWORD *)(v9[6] + 8LL) : ((__int64 (*)(void))v10)();
    if ( a1 == v11 )
      return 1;
    v12 = *(_BYTE *)(v2 + 8);
    if ( v12 == 1 )
      return 1;
    if ( v12 == 9 )
    {
      v4 = *(unsigned __int8 *)(v2 + 160);
      if ( !(_BYTE)v4 )
        return v4;
      if ( (unsigned __int8)(**(_BYTE **)(v2 + 136) - 61) <= 1u )
      {
        v13 = *(char **)(v2 + 48);
        v14 = 8LL * *(unsigned int *)(v2 + 56);
        v15 = &v13[v14];
        v16 = v14 >> 5;
        if ( v16 )
        {
          v17 = &v13[32 * v16];
          while ( (unsigned __int8)sub_2C46F30(*(_QWORD *)v13) )
          {
            if ( !(unsigned __int8)sub_2C46F30(*((_QWORD *)v13 + 1)) )
              goto LABEL_42;
            if ( !(unsigned __int8)sub_2C46F30(*((_QWORD *)v13 + 2)) )
              goto LABEL_43;
            if ( !(unsigned __int8)sub_2C46F30(*((_QWORD *)v13 + 3)) )
              goto LABEL_44;
            v13 += 32;
            if ( v17 == v13 )
              goto LABEL_55;
          }
          goto LABEL_22;
        }
LABEL_55:
        v24 = v15 - v13;
        if ( v15 - v13 != 16 )
        {
          if ( v24 != 24 )
          {
            if ( v24 == 8 )
              goto LABEL_49;
            return v4;
          }
          goto LABEL_51;
        }
        goto LABEL_53;
      }
      return 0;
    }
    if ( v12 != 10 && v12 != 16 )
      return 0;
    a1 = **(_QWORD **)(v2 + 48);
  }
  v19 = sub_2BF0490(a1);
  if ( *(_BYTE *)(v19 + 8) != 4 || *(_BYTE *)(v19 + 160) != 77 )
  {
    v13 = *(char **)(v2 + 48);
    v20 = 8LL * *(unsigned int *)(v2 + 56);
    v15 = &v13[v20];
    v21 = v20 >> 3;
    v22 = v20 >> 5;
    if ( v22 )
    {
      v23 = &v13[32 * v22];
      while ( (unsigned __int8)sub_2C46F30(*(_QWORD *)v13) )
      {
        if ( !(unsigned __int8)sub_2C46F30(*((_QWORD *)v13 + 1)) )
        {
LABEL_42:
          LOBYTE(v4) = v15 == v13 + 8;
          return v4;
        }
        if ( !(unsigned __int8)sub_2C46F30(*((_QWORD *)v13 + 2)) )
        {
LABEL_43:
          LOBYTE(v4) = v15 == v13 + 16;
          return v4;
        }
        if ( !(unsigned __int8)sub_2C46F30(*((_QWORD *)v13 + 3)) )
        {
LABEL_44:
          LOBYTE(v4) = v15 == v13 + 24;
          return v4;
        }
        v13 += 32;
        if ( v13 == v23 )
        {
          v21 = (v15 - v13) >> 3;
          goto LABEL_46;
        }
      }
LABEL_22:
      LOBYTE(v4) = v15 == v13;
      return v4;
    }
LABEL_46:
    if ( v21 != 2 )
    {
      if ( v21 != 3 )
      {
        if ( v21 == 1 )
        {
LABEL_49:
          v4 = sub_2C46F30(*(_QWORD *)v13);
          if ( !(_BYTE)v4 )
            goto LABEL_22;
        }
        return v4;
      }
LABEL_51:
      if ( !(unsigned __int8)sub_2C46F30(*(_QWORD *)v13) )
        goto LABEL_22;
      v13 += 8;
    }
LABEL_53:
    if ( !(unsigned __int8)sub_2C46F30(*(_QWORD *)v13) )
      goto LABEL_22;
    v13 += 8;
    goto LABEL_49;
  }
  return 0;
}
