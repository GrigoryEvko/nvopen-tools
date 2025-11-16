// Function: sub_8F0220
// Address: 0x8f0220
//
_BYTE *__fastcall sub_8F0220(_BYTE *a1, int *a2)
{
  _BYTE *v3; // rbx
  int v4; // eax
  unsigned __int64 v5; // rsi
  unsigned __int64 v6; // r11
  char v7; // r9
  char v8; // r10
  __int64 i; // rax
  __int64 v10; // rdx
  _BYTE *v11; // rdx
  _BYTE *result; // rax
  _BYTE *v13; // rdx
  unsigned int v14; // r8d
  int v15; // eax
  int v16; // eax
  _QWORD v17[3]; // [rsp+8h] [rbp-18h] BYREF

  v3 = a1;
  v4 = *a2;
  if ( *a2 == 2 )
  {
    v14 = a2[7];
    v5 = *(_QWORD *)(a2 + 3);
    v15 = a2[2];
    v17[0] = v5;
    if ( v15 < (int)(-16381 - v14) )
    {
      *a2 = 6;
      LOBYTE(v5) = 0;
      LOBYTE(v6) = 0;
      v7 = 0;
      v17[0] = 0;
      v8 = 0;
      goto LABEL_4;
    }
    if ( v15 > 0x4000 )
    {
      *a2 = 4;
      LODWORD(v17[0]) = 0;
      WORD2(v17[0]) = 0;
      BYTE6(v17[0]) = 0;
      goto LABEL_3;
    }
    v16 = v15 + 16382;
    if ( v16 > 0 )
    {
      v8 = v16;
      v7 = BYTE1(v16);
      v6 = HIBYTE(v5);
      goto LABEL_4;
    }
    sub_8EE610((char *)v17, v14, 1 - v16, 0);
    v4 = *a2;
    if ( *a2 == 2 )
    {
      LOBYTE(v6) = HIBYTE(v17[0]);
      LOBYTE(v5) = v17[0];
      v7 = 0;
      v8 = 0;
      goto LABEL_4;
    }
  }
  v17[0] = 0;
  switch ( v4 )
  {
    case 4:
LABEL_3:
      HIBYTE(v17[0]) = 0x80;
      LOBYTE(v5) = 0;
      LOBYTE(v6) = 0x80;
      v7 = 127;
      v8 = -1;
      break;
    case 6:
      LOBYTE(v5) = 0;
      LOBYTE(v6) = 0;
      v7 = 0;
      v8 = 0;
      break;
    case 3:
      LOBYTE(v17[0]) = 1;
      LOBYTE(v5) = 1;
      LOBYTE(v6) = -64;
      v7 = 127;
      HIBYTE(v17[0]) = -64;
      v8 = -1;
      break;
    default:
      sub_721090();
  }
LABEL_4:
  for ( i = 0; ; LOBYTE(v5) = *((_BYTE *)v17 + i) )
  {
    v10 = 9 - i;
    if ( dword_4F07580 )
      v10 = i;
    ++i;
    a1[v10] = v5;
    if ( i == 7 )
      break;
  }
  a1[(-(__int64)(dword_4F07580 == 0) & 0xFFFFFFFFFFFFFFFBLL) + 7] = 0;
  if ( !dword_4F07580 )
    i = 2;
  v11 = a1;
  a1[i] |= v6;
  a1[(-(__int64)(dword_4F07580 == 0) & 0xFFFFFFFFFFFFFFF9LL) + 8] = 0;
  a1[(-(__int64)(dword_4F07580 == 0) & 0xFFFFFFFFFFFFFFF9LL) + 8] |= v8;
  result = a1 + 9;
  if ( dword_4F07580 )
    v11 = a1 + 9;
  *v11 &= 0x80u;
  v13 = a1;
  if ( dword_4F07580 )
    v13 = a1 + 9;
  *v13 |= v7;
  if ( a2[1] )
  {
    if ( dword_4F07580 )
      v3 = a1 + 9;
    *v3 |= 0x80u;
  }
  else
  {
    if ( dword_4F07580 )
      v3 = a1 + 9;
    *v3 &= ~0x80u;
  }
  return result;
}
