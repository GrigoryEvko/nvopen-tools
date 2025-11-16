// Function: sub_7CA090
// Address: 0x7ca090
//
__int64 sub_7CA090()
{
  _BYTE *v0; // rax
  _QWORD *v1; // rax
  char *v2; // rax
  __int64 v3; // rbx
  int v4; // r8d
  _BOOL4 v5; // eax
  __int64 v6; // rdx
  unsigned int *v7; // rsi
  unsigned __int64 v8; // rcx
  __int64 v9; // r9
  unsigned int v10; // r8d
  __int64 v11; // rax
  _DWORD *v12; // rcx
  _DWORD *v13; // rdx
  unsigned int i; // eax
  unsigned int *v15; // kr00_8
  _DWORD *v16; // rdx
  int v17; // eax
  unsigned __int8 *v18; // rsi
  int v19; // ecx
  unsigned __int8 v20; // r8
  int v21; // r10d
  __int64 **v22; // rax
  __int64 result; // rax

  v0 = (_BYTE *)sub_822BE0(3002);
  qword_4F06498 = v0 + 1;
  *v0 = 32;
  qword_4F06490 = qword_4F06498 + 3000LL;
  v1 = (_QWORD *)sub_822BE0(24000);
  dword_4F084E8 = 0;
  qword_4F17F90 = 0;
  qword_4F06488 = v1;
  qword_4F17F88 = 0;
  *(_DWORD *)&word_4F06480 = 0;
  if ( qword_4D04908 )
  {
    v2 = (char *)sub_822BE0(3000);
    dword_4F17F7C = 0;
    qword_4F17F90 = v2;
    qword_4F17F88 = (__int64)(v2 + 3000);
    qword_4F17F80 = (__int64)v2;
  }
  qword_4F17FE0 = 0;
  v3 = -128;
  dword_4F17FDC = 0;
  dword_4F084D8 = 0;
  do
  {
    v4 = isalpha((unsigned __int8)v3);
    v5 = 1;
    if ( !v4 )
      v5 = (unsigned int)(unsigned __int8)v3 - 48 <= 9;
    v6 = v3++;
    dword_4F05DC0[v6 + 128] = v5;
    dword_4F04DC0[v6 + 128] = v5;
  }
  while ( v3 != 128 );
  dword_4F05DC0[223] = 1;
  if ( unk_4D04748 )
    dword_4F05DC0[164] = 1;
  *(_QWORD *)&dword_4F05DC0[219] = 0;
  v7 = dword_4F059C0;
  v8 = 4294967256LL;
  *(_QWORD *)&dword_4F04DC0[221] = 0x100000001LL;
  v9 = 0x7FFFFF77BCA5LL;
  *(_QWORD *)&dword_4F04DC0[251] = 0x100000001LL;
  *(_QWORD *)&dword_4F04DC0[253] = 0x100000001LL;
  *(_QWORD *)&dword_4F04DC0[186] = 0x100000001LL;
  *(_QWORD *)&dword_4F04DC0[188] = 0x100000001LL;
  *(_QWORD *)&dword_4F04DC0[190] = 0x100000001LL;
  *(_QWORD *)&dword_4F04DC0[170] = 0x100000001LL;
  *(_QWORD *)&dword_4F04DC0[172] = 0x100000001LL;
  *(_QWORD *)&dword_4F04DC0[174] = 0x100000001LL;
  *(_QWORD *)&dword_4F04DC0[161] = 0x100000001LL;
  *(_QWORD *)&dword_4F04DC0[165] = 0x100000001LL;
  *(_QWORD *)&dword_4F05DC0[221] = 0;
  *(_QWORD *)&dword_4F05DC0[251] = 0;
  v10 = dword_4D0432C;
  v11 = -128;
  *(_QWORD *)&dword_4F05DC0[253] = 0;
  dword_4F04DC0[219] = 1;
  dword_4F04DC0[223] = 1;
  dword_4F04DC0[163] = 1;
  dword_4F04DC0[167] = 1;
  while ( (int)v11 < 0 )
  {
    dword_4F059C0[(unsigned __int8)v11] = (unsigned __int8)v8 <= 0x2Eu && _bittest64(&v9, v8)
                                       || (unsigned __int8)(v11 + 40) <= 0x1Eu
                                       || (unsigned __int8)v11 > 0xF7u;
    if ( v10 )
      dword_4F05DC0[v11 + 128] = 0;
LABEL_11:
    ++v11;
    v8 = (unsigned int)(v8 + 1);
  }
  dword_4F059C0[(unsigned __int8)v11] = dword_4F05DC0[v11 + 128];
  if ( (_DWORD)v11 != 127 )
    goto LABEL_11;
  v12 = dword_4F055C0;
  v13 = dword_4F051C0;
  for ( i = -128; i != 128; ++i )
  {
    v15 = v7;
    v7 = (unsigned int *)i;
    switch ( i )
    {
      case 0u:
      case 0x20u:
      case 0x22u:
      case 0x23u:
      case 0x24u:
      case 0x25u:
      case 0x26u:
      case 0x28u:
      case 0x29u:
      case 0x2Au:
      case 0x2Bu:
      case 0x2Cu:
      case 0x2Du:
      case 0x2Fu:
      case 0x3Au:
      case 0x3Bu:
      case 0x3Cu:
      case 0x3Du:
      case 0x3Eu:
      case 0x3Fu:
      case 0x5Bu:
      case 0x5Du:
      case 0x5Eu:
      case 0x7Bu:
      case 0x7Cu:
      case 0x7Du:
      case 0x7Eu:
        *v12 = 1;
        *v13 = 1;
        break;
      case 0x27u:
      case 0x2Eu:
        *v12 = 1;
        *v13 = 0;
        break;
      default:
        v7 = v15;
        *v12 = 0;
        *v13 = 0;
        break;
    }
    ++v12;
    ++v13;
  }
  v16 = dword_4F05DC0;
  v17 = -128;
  v18 = byte_4F04C80;
  v19 = -127;
  v20 = 2 * (dword_4F077C4 == 2) + 1;
  if ( dword_4F05DC0[0] )
    goto LABEL_34;
LABEL_25:
  if ( v17 == 46 )
  {
LABEL_34:
    while ( 1 )
    {
      *v18 = 2;
      if ( v19 == 128 )
        break;
LABEL_33:
      v21 = v16[1];
      v17 = v19;
      ++v16;
      ++v18;
      ++v19;
      if ( !v21 )
        goto LABEL_25;
    }
  }
  else
  {
    if ( v17 > 12 )
    {
      switch ( v17 )
      {
        case ' ':
        case '(':
        case ')':
        case ',':
        case ';':
        case '?':
        case '[':
        case ']':
        case '{':
        case '}':
        case '~':
          goto LABEL_41;
        case ':':
          byte_4F04C80[186] = v20;
          goto LABEL_33;
        default:
          goto LABEL_31;
      }
    }
    if ( v17 > 8 )
LABEL_41:
      *v18 = 1;
    else
LABEL_31:
      *v18 = 3;
    if ( v19 != 128 )
      goto LABEL_33;
  }
  sub_708C50();
  v22 = sub_7ABDF0("c:C:cpp:CPP:cxx:CXX:cc");
  qword_4F084F0 = 0;
  qword_4F084F8 = (__int64)v22;
  if ( qword_4D042C0 && *qword_4D042C0 )
    qword_4F084F0 = (__int64)sub_7ABDF0(qword_4D042C0);
  else
    sub_7ABD00((__int64 ***)&qword_4F084F0, byte_3F871B3, 0);
  if ( unk_4D04508 )
    sub_8539C0(&off_4B6F1C0);
  sub_8D0840(&unk_4F04D84, 4, 0);
  sub_8D0840(&qword_4F061C8, 8, 0);
  sub_8D0840(&qword_4F061C0, 8, 0);
  sub_8D0840(word_4F06418, 2, 0);
  sub_8D0840(&unk_4D03E88, 8, 0);
  sub_8D0840(xmmword_4F06300, 208, 0);
  sub_8D0840(&dword_4F063F8, 8, 0);
  sub_8D0840(&qword_4F063F0, 8, 0);
  sub_8D0840(&qword_4F06410, 8, 0);
  sub_8D0840(&qword_4F06408, 8, 0);
  sub_8D0840(&unk_4F06400, 8, 0);
  sub_8D0840(&qword_4F08560, 8, 0);
  sub_8D0840(&qword_4F08538, 8, 0);
  sub_8D0840(&dword_4F061FC, 4, 0);
  sub_8D0840(&unk_4F061F8, 4, 0);
  sub_8D0840(&unk_4F061F0, 8, 0);
  sub_8D0840(dword_4F06650, 4, 0);
  sub_8D0840(&unk_4F06640, 8, 0);
  result = sub_881A70(0xFFFFFFFFLL, 1024, 6, 7);
  qword_4F08500 = result;
  return result;
}
