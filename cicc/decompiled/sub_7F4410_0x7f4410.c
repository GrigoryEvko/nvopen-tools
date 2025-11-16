// Function: sub_7F4410
// Address: 0x7f4410
//
void __fastcall sub_7F4410(__int64 a1, __m128i *a2)
{
  int v3; // edx
  __int64 *i; // rbx
  __int64 v5; // rax
  _QWORD *j; // rbx
  __int64 v7; // rdi
  __int64 v8; // rax
  char v9; // al
  unsigned __int8 v10; // al
  _QWORD *k; // r13
  __int64 v12; // rbx
  __int64 v13; // rdi
  int v14; // edx
  __int64 v15; // rdi

  if ( (*(_BYTE *)(a1 - 8) & 8) != 0 )
    return;
  if ( (*(_BYTE *)(a1 + 198) & 0x40) == 0 )
  {
    v3 = *(_DWORD *)(a1 + 64);
    *(_BYTE *)(a1 - 8) |= 8u;
    if ( !v3 )
      goto LABEL_5;
    goto LABEL_4;
  }
  sub_8E5740();
  v14 = *(_DWORD *)(a1 + 64);
  *(_BYTE *)(a1 - 8) |= 8u;
  if ( v14 )
LABEL_4:
    *(_QWORD *)dword_4F07508 = *(_QWORD *)(a1 + 64);
LABEL_5:
  if ( (*(_BYTE *)(a1 + 88) & 0x70) == 0x20 )
    *(_BYTE *)(a1 + 88) = *(_BYTE *)(a1 + 88) & 0x8F | 0x30;
  sub_7E2D70(a1);
  if ( (unsigned __int8)(*(_BYTE *)(a1 + 174) - 1) <= 1u )
    sub_7FA1F0(a1);
  sub_7EA690(*(_QWORD *)(a1 + 152), a2);
  for ( i = *(__int64 **)(a1 + 240); i; i = (__int64 *)*i )
  {
    v10 = *((_BYTE *)i + 8);
    if ( v10 == 1 )
    {
      sub_7EB190(i[4], a2);
    }
    else if ( v10 <= 1u )
    {
      sub_7EA690(i[4], a2);
    }
    else if ( (unsigned __int8)(v10 - 2) > 1u )
    {
      sub_721090();
    }
  }
  v5 = *(_QWORD *)(a1 + 264);
  if ( v5 )
  {
    while ( *(_BYTE *)(v5 + 140) == 12 )
      v5 = *(_QWORD *)(v5 + 160);
    for ( j = **(_QWORD ***)(v5 + 168); j; j = (_QWORD *)*j )
    {
      v7 = j[5];
      if ( v7 )
      {
        sub_7E99E0(v7);
        j[5] = 0;
      }
    }
  }
  if ( *(char *)(a1 + 192) < 0 )
  {
    if ( *(_BYTE *)(a1 + 172) )
      goto LABEL_20;
    if ( (*(_BYTE *)(a1 + 203) & 0x40) != 0 )
    {
      if ( (*(_BYTE *)(a1 + 195) & 3) != 1 )
        goto LABEL_20;
      goto LABEL_53;
    }
    sub_7E5120(a1);
  }
  if ( (*(_BYTE *)(a1 + 195) & 3) != 1 || *(_BYTE *)(a1 + 172) )
  {
LABEL_20:
    v8 = *(_QWORD *)(a1 + 272);
    if ( !v8 )
      goto LABEL_22;
    goto LABEL_21;
  }
LABEL_53:
  if ( (*(_QWORD *)(a1 + 200) & 0x20040000000LL) == 0 )
    goto LABEL_62;
  if ( *(_BYTE *)(a1 + 174) != 6 )
    goto LABEL_20;
  v15 = *(_QWORD *)(a1 + 176);
  if ( v15 )
  {
    if ( (*(_BYTE *)(v15 + 205) & 2) != 0 )
    {
LABEL_62:
      sub_7E5120(a1);
      goto LABEL_20;
    }
  }
  v8 = *(_QWORD *)(a1 + 272);
  if ( !v8 )
    goto LABEL_60;
LABEL_21:
  if ( (*(_BYTE *)(v8 + 203) & 0x40) == 0 && *(_DWORD *)(v8 + 160) )
    sub_8076F0(a1);
LABEL_22:
  v9 = *(_BYTE *)(a1 + 174);
  if ( (unsigned __int8)(v9 - 1) <= 1u )
  {
    if ( *(_QWORD *)(a1 + 320) )
      goto LABEL_24;
    sub_7FE940(a1, 0);
    for ( k = *(_QWORD **)(a1 + 176); k; k = (_QWORD *)*k )
    {
      v12 = k[1];
      if ( sub_736A10(v12) )
      {
        sub_7E99A0(v12);
        *(_BYTE *)(v12 + 172) = 0;
        sub_7604D0(v12, 0xBu);
      }
      do
      {
        v13 = v12;
        v12 = *(_QWORD *)(v12 + 112);
        if ( (*(_DWORD *)(a1 + 192) & 0x8000400) == 0 )
          sub_7F4410(v13);
      }
      while ( v12 );
    }
    v9 = *(_BYTE *)(a1 + 174);
  }
  if ( v9 == 6 )
  {
    v15 = *(_QWORD *)(a1 + 176);
LABEL_60:
    sub_7FCF80(v15, a1, 0);
  }
LABEL_24:
  if ( (*(_BYTE *)(a1 + 201) & 1) != 0 )
    sub_807D50(a1);
  sub_808590(a1);
}
