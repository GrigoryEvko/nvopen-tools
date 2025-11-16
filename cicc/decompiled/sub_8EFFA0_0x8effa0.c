// Function: sub_8EFFA0
// Address: 0x8effa0
//
_BYTE *__fastcall sub_8EFFA0(_BYTE *a1, int *a2)
{
  _BYTE *v3; // rbx
  int v4; // eax
  char v5; // si
  char v6; // r11
  int v7; // r9d
  char v8; // r10
  __int64 i; // rax
  __int64 v10; // rdx
  _BYTE *v11; // rdx
  _BYTE *result; // rax
  _BYTE *v13; // rdx
  char v14; // r11
  __int16 v15; // ax
  int v16; // eax
  unsigned int v17; // esi
  int v18; // r9d
  int v19; // [rsp+9h] [rbp-17h] BYREF
  __int16 v20; // [rsp+Dh] [rbp-13h]
  char v21; // [rsp+Fh] [rbp-11h]

  v3 = a1;
  v4 = *a2;
  if ( *a2 == 2 )
  {
    v14 = *((_BYTE *)a2 + 18);
    v19 = a2[3];
    v15 = *((_WORD *)a2 + 8);
    v21 = v14;
    v20 = v15;
    v16 = a2[2];
    v17 = a2[7];
    if ( v16 < (int)(-1021 - v17) )
    {
      *a2 = 6;
      v6 = 0;
      LOBYTE(v7) = 0;
      v20 = 0;
      v8 = 0;
      v5 = 0;
      v19 = 0;
      v21 = 0;
      goto LABEL_4;
    }
    if ( v16 > 1024 )
    {
      *a2 = 4;
      v5 = 0;
      v6 = 0;
      v20 = 0;
      LOBYTE(v7) = 127;
      v8 = -16;
      v19 = 0;
      v21 = 0;
      goto LABEL_4;
    }
    v18 = v16 + 1022;
    if ( v16 + 1022 > 0 )
    {
      v5 = v19;
      v7 = v18 >> 4;
      v6 = v14 & 0xF;
      v8 = 16 * (v16 - 2);
      goto LABEL_4;
    }
    sub_8EE610((char *)&v19, v17, 1 - v18, 0);
    v4 = *a2;
    if ( *a2 == 2 )
    {
      v5 = v19;
      LOBYTE(v7) = 0;
      v8 = 0;
      v6 = v21 & 0xF;
      goto LABEL_4;
    }
  }
  v19 = 0;
  v20 = 0;
  v21 = 0;
  switch ( v4 )
  {
    case 4:
      v5 = 0;
      v6 = 0;
      LOBYTE(v7) = 127;
      v8 = -16;
      break;
    case 6:
      v5 = 0;
      v6 = 0;
      LOBYTE(v7) = 0;
      v8 = 0;
      break;
    case 3:
      LOBYTE(v19) = 1;
      v5 = 1;
      v6 = 8;
      LOBYTE(v7) = 127;
      v21 = 8;
      v8 = -16;
      break;
    default:
      sub_721090();
  }
LABEL_4:
  for ( i = 0; ; v5 = *((_BYTE *)&v19 + i) )
  {
    v10 = 7 - i;
    if ( dword_4F07580 )
      v10 = i;
    ++i;
    a1[v10] = v5;
    if ( i == 6 )
      break;
  }
  a1[(-(__int64)(dword_4F07580 == 0) & 0xFFFFFFFFFFFFFFFBLL) + 6] &= 0xF0u;
  a1[(-(__int64)(dword_4F07580 == 0) & 0xFFFFFFFFFFFFFFFBLL) + 6] |= v6;
  a1[(-(__int64)(dword_4F07580 == 0) & 0xFFFFFFFFFFFFFFFBLL) + 6] &= 0xFu;
  if ( !dword_4F07580 )
    i = 1;
  v11 = a1;
  a1[i] |= v8;
  result = a1 + 7;
  if ( dword_4F07580 )
    v11 = a1 + 7;
  *v11 &= 0x80u;
  v13 = a1;
  if ( dword_4F07580 )
    v13 = a1 + 7;
  *v13 |= v7;
  if ( a2[1] )
  {
    if ( dword_4F07580 )
      v3 = a1 + 7;
    *v3 |= 0x80u;
  }
  else
  {
    if ( dword_4F07580 )
      v3 = a1 + 7;
    *v3 &= ~0x80u;
  }
  return result;
}
