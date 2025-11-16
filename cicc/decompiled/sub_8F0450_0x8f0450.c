// Function: sub_8F0450
// Address: 0x8f0450
//
_BYTE *__fastcall sub_8F0450(_BYTE *a1, int *a2)
{
  _BYTE *v3; // rbx
  int v4; // eax
  char v5; // si
  char v6; // r11
  char v7; // r9
  char v8; // r10
  __int64 i; // rax
  __int64 v10; // rdx
  _BYTE *v11; // rdx
  _BYTE *result; // rax
  _BYTE *v13; // rdx
  int v14; // eax
  unsigned int v15; // esi
  int v16; // eax
  __int64 v17; // [rsp+1h] [rbp-1Fh] BYREF
  int v18; // [rsp+9h] [rbp-17h]
  __int16 v19; // [rsp+Dh] [rbp-13h]
  char v20; // [rsp+Fh] [rbp-11h]

  v3 = a1;
  v4 = *a2;
  if ( *a2 == 2 )
  {
    v17 = *(_QWORD *)(a2 + 3);
    v18 = a2[5];
    v19 = *((_WORD *)a2 + 12);
    v20 = *((_BYTE *)a2 + 26);
    v14 = a2[2];
    v15 = a2[7];
    if ( v14 < (int)(-16381 - v15) )
    {
      *a2 = 6;
      v6 = 0;
      v7 = 0;
      v19 = 0;
      v8 = 0;
      v5 = 0;
      v17 = 0;
      v18 = 0;
      v20 = 0;
      goto LABEL_4;
    }
    if ( v14 > 0x4000 )
    {
      *a2 = 4;
      v5 = 0;
      v6 = 0;
      v19 = 0;
      v7 = 127;
      v8 = -1;
      v17 = 0;
      v18 = 0;
      v20 = 0;
      goto LABEL_4;
    }
    v16 = v14 + 16382;
    if ( v16 > 0 )
    {
      v8 = v16;
      v6 = HIBYTE(v19);
      v5 = v17;
      v7 = BYTE1(v16);
      goto LABEL_4;
    }
    sub_8EE610((char *)&v17, v15, 1 - v16, 0);
    v4 = *a2;
    if ( *a2 == 2 )
    {
      v6 = HIBYTE(v19);
      v5 = v17;
      v7 = 0;
      v8 = 0;
      goto LABEL_4;
    }
  }
  v17 = 0;
  v18 = 0;
  v19 = 0;
  v20 = 0;
  switch ( v4 )
  {
    case 4:
      v5 = 0;
      v6 = 0;
      v7 = 127;
      v8 = -1;
      break;
    case 6:
      v5 = 0;
      v6 = 0;
      v7 = 0;
      v8 = 0;
      break;
    case 3:
      LOBYTE(v17) = 1;
      v5 = 1;
      v6 = 0x80;
      v7 = 127;
      HIBYTE(v19) = 0x80;
      v8 = -1;
      break;
    default:
      sub_721090();
  }
LABEL_4:
  for ( i = 0; ; v5 = *((_BYTE *)&v17 + i) )
  {
    v10 = 15 - i;
    if ( dword_4F07580 )
      v10 = i;
    ++i;
    a1[v10] = v5;
    if ( i == 13 )
      break;
  }
  a1[(-(__int64)(dword_4F07580 == 0) & 0xFFFFFFFFFFFFFFF5LL) + 13] = 0;
  if ( !dword_4F07580 )
    i = 2;
  v11 = a1;
  a1[i] |= v6;
  a1[(-(__int64)(dword_4F07580 == 0) & 0xFFFFFFFFFFFFFFF3LL) + 14] = 0;
  a1[(-(__int64)(dword_4F07580 == 0) & 0xFFFFFFFFFFFFFFF3LL) + 14] |= v8;
  result = a1 + 15;
  if ( dword_4F07580 )
    v11 = a1 + 15;
  *v11 &= 0x80u;
  v13 = a1;
  if ( dword_4F07580 )
    v13 = a1 + 15;
  *v13 |= v7;
  if ( a2[1] )
  {
    if ( dword_4F07580 )
      v3 = a1 + 15;
    *v3 |= 0x80u;
  }
  else
  {
    if ( dword_4F07580 )
      v3 = a1 + 15;
    *v3 &= ~0x80u;
  }
  return result;
}
