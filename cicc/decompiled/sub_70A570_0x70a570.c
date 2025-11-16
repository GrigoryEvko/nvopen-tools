// Function: sub_70A570
// Address: 0x70a570
//
_QWORD *__fastcall sub_70A570(__int64 a1, __int64 a2, _QWORD *a3, _DWORD *a4)
{
  unsigned __int8 *v4; // r15
  unsigned __int8 i; // al
  unsigned __int8 v6; // r14
  __int64 v7; // rbx
  int v8; // r12d
  int v9; // eax
  _BOOL4 v10; // esi
  int v11; // ecx
  int v12; // edx
  char v13; // cl
  int v14; // edx
  _BYTE *v15; // rdi
  __int64 v16; // r8
  int v17; // ecx
  __int64 v18; // rdx
  int v23; // [rsp+20h] [rbp-40h]
  _BOOL4 v24; // [rsp+24h] [rbp-3Ch]
  _BOOL4 v25; // [rsp+28h] [rbp-38h]
  int v26; // [rsp+2Ch] [rbp-34h]

  v4 = (unsigned __int8 *)(a1 + 2);
  *a4 = 0;
  sub_70A200(a2);
  for ( i = *(_BYTE *)(a1 + 2); i == 48; ++v4 )
    i = v4[1];
  if ( i == 46 )
  {
    v6 = *++v4;
    if ( v6 == 48 )
    {
      v7 = 0;
      do
      {
        v6 = *++v4;
        v7 -= 4;
      }
      while ( v6 == 48 );
      v26 = 1;
    }
    else
    {
      v26 = 1;
      v7 = 0;
    }
  }
  else
  {
    v26 = 0;
    v6 = *v4;
    v7 = 0;
  }
  v24 = 0;
  v8 = 0;
  v25 = 0;
  v23 = 0;
  while ( 1 )
  {
    v9 = isxdigit(v6);
    if ( !v9 )
      break;
    if ( v6 == 46 )
      goto LABEL_7;
    if ( v25 )
    {
      v10 = v24;
      if ( v6 != 48 )
        v10 = v25;
      v24 = v10;
    }
    else
    {
      v11 = 48;
      if ( (unsigned int)v6 - 48 > 9 )
        v11 = islower(v6) == 0 ? 55 : 87;
      v12 = (char)v6 - v11;
      v13 = 7 - v8++;
      *(_DWORD *)(a2 + 4LL * v23) |= v12 << (4 * v13);
      if ( v8 == 8 )
      {
        ++v23;
        v8 = 0;
        v25 = v23 > 3;
      }
    }
    if ( !v26 )
      v7 += 4;
LABEL_8:
    v6 = *++v4;
  }
  if ( v6 == 46 )
  {
LABEL_7:
    v26 = 1;
    goto LABEL_8;
  }
  if ( (v6 & 0xDF) != 0x50 )
    goto LABEL_34;
  LOBYTE(v14) = v4[1];
  if ( (_BYTE)v14 == 45 )
  {
    v15 = v4 + 2;
    v14 = v4[2];
    if ( (unsigned int)(v14 - 48) > 9 )
      goto LABEL_34;
    v9 = 1;
LABEL_27:
    v16 = 0;
    do
    {
      if ( unk_4F06928 < v16 )
        *a4 = 1;
      else
        v16 = (char)v14 - 48 + 10 * v16;
      v17 = (unsigned __int8)*++v15;
      LOBYTE(v14) = v17;
    }
    while ( (unsigned int)(v17 - 48) <= 9 );
    v18 = v7 + v16;
    v7 -= v16;
    if ( !v9 )
      v7 = v18;
    goto LABEL_34;
  }
  if ( (_BYTE)v14 == 43 )
  {
    v15 = v4 + 2;
    v14 = v4[2];
    if ( (unsigned int)(v14 - 48) > 9 )
      goto LABEL_34;
    goto LABEL_27;
  }
  if ( (unsigned int)(unsigned __int8)v14 - 48 <= 9 )
  {
    v15 = v4 + 1;
    goto LABEL_27;
  }
LABEL_34:
  if ( v24 )
    *(_DWORD *)(a2 + 16) = 1;
  *a3 = v7;
  return a3;
}
