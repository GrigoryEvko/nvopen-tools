// Function: sub_8EFD90
// Address: 0x8efd90
//
unsigned int *__fastcall sub_8EFD90(char *a1, int *a2)
{
  int v4; // eax
  char v5; // r9
  char v6; // cl
  char v7; // di
  int v8; // esi
  char v9; // r8
  unsigned int *result; // rax
  char *v11; // r10
  char *v12; // rcx
  char *v13; // rcx
  unsigned int v14; // r8d
  char v15; // di
  int v16; // eax
  int v17; // esi
  __int16 v18; // [rsp+Dh] [rbp-13h] BYREF
  char v19; // [rsp+Fh] [rbp-11h]

  v4 = *a2;
  if ( *a2 == 2 )
  {
    v14 = a2[7];
    v15 = *((_BYTE *)a2 + 14);
    v18 = *((_WORD *)a2 + 6);
    v16 = a2[2];
    v19 = v15;
    if ( v16 < (int)(-125 - v14) )
    {
      *a2 = 6;
      v18 = 0;
      v19 = 0;
LABEL_24:
      v5 = 0;
      v6 = 0;
      v7 = 0;
      LOBYTE(v8) = 0;
      v9 = 0;
      goto LABEL_4;
    }
    if ( v16 > 128 )
    {
      *a2 = 4;
      v18 = 0;
      v19 = 0;
      goto LABEL_3;
    }
    v17 = v16 + 126;
    if ( v16 + 126 > 0 )
    {
      v6 = v18;
      v5 = HIBYTE(v18);
      v8 = v17 >> 1;
      v9 = ((_BYTE)v16 + 126) << 7;
      v7 = v15 & 0x7F;
      goto LABEL_4;
    }
    sub_8EE610((char *)&v18, v14, 1 - v17, 0);
    v4 = *a2;
    if ( *a2 == 2 )
    {
      v6 = v18;
      LOBYTE(v8) = 0;
      v9 = 0;
      v5 = HIBYTE(v18);
      v7 = v19 & 0x7F;
      goto LABEL_4;
    }
  }
  v19 = 0;
  v18 = 0;
  if ( v4 == 4 )
  {
LABEL_3:
    v5 = 0;
    v6 = 0;
    v7 = 0;
    LOBYTE(v8) = 127;
    v9 = 0x80;
    goto LABEL_4;
  }
  if ( v4 == 6 )
    goto LABEL_24;
  if ( v4 != 3 )
    sub_721090();
  LOBYTE(v18) = 1;
  v5 = 0;
  v6 = 1;
  v7 = 64;
  v19 = 64;
  LOBYTE(v8) = 127;
  v9 = 0x80;
LABEL_4:
  result = &dword_4F07580;
  v11 = a1;
  if ( !dword_4F07580 )
    v11 = a1 + 3;
  *v11 = v6;
  a1[(dword_4F07580 == 0) + 1] = v5;
  a1[-(dword_4F07580 == 0) + 2] &= 0x80u;
  a1[-(dword_4F07580 == 0) + 2] |= v7;
  a1[-(dword_4F07580 == 0) + 2] &= ~0x80u;
  a1[-(dword_4F07580 == 0) + 2] |= v9;
  v12 = a1;
  if ( dword_4F07580 )
    v12 = a1 + 3;
  *v12 &= 0x80u;
  v13 = a1;
  if ( dword_4F07580 )
    v13 = a1 + 3;
  *v13 |= v8;
  if ( a2[1] )
  {
    result = (unsigned int *)dword_4F07580;
    if ( dword_4F07580 )
      a1 += 3;
    *a1 |= 0x80u;
  }
  else
  {
    if ( dword_4F07580 )
      a1 += 3;
    *a1 &= ~0x80u;
  }
  return result;
}
