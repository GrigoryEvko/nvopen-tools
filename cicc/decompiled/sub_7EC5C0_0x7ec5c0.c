// Function: sub_7EC5C0
// Address: 0x7ec5c0
//
__int64 __fastcall sub_7EC5C0(__int64 a1, __m128i *a2)
{
  int v2; // ecx
  __int64 *i; // rbx
  char v4; // al
  int v5; // ebx
  __int64 result; // rax
  unsigned __int8 v7; // al
  const char *v8; // rsi

  v2 = *(_DWORD *)(a1 + 64);
  *(_BYTE *)(a1 - 8) |= 8u;
  if ( v2 )
    *(_QWORD *)dword_4F07508 = *(_QWORD *)(a1 + 64);
  if ( (*(_BYTE *)(a1 + 88) & 0x70) == 0x20 )
    *(_BYTE *)(a1 + 88) = *(_BYTE *)(a1 + 88) & 0x8F | 0x30;
  sub_7E2D70(a1);
  sub_7EAF80(*(_QWORD *)(a1 + 120), a2);
  if ( (*(_BYTE *)(a1 + 170) & 0x10) != 0 )
  {
    for ( i = **(__int64 ***)(a1 + 216); i; i = (__int64 *)*i )
    {
      v7 = *((_BYTE *)i + 8);
      if ( v7 == 1 )
      {
        sub_7EB190(i[4], a2);
      }
      else if ( v7 <= 1u )
      {
        sub_7EA690(i[4], a2);
      }
      else if ( (unsigned __int8)(v7 - 2) > 1u )
      {
        sub_721090();
      }
    }
  }
  if ( (*(_BYTE *)(a1 + 169) & 0x40) != 0 && *(_BYTE *)(a1 + 136) == 5 )
  {
    *(_BYTE *)(a1 + 136) = 3;
    goto LABEL_12;
  }
  v4 = *(_BYTE *)(a1 + 177);
  if ( !dword_4F0696C || *(_BYTE *)(a1 + 136) )
  {
    if ( v4 || (*(_BYTE *)(a1 + 136) & 0xFD) != 0 )
      goto LABEL_32;
  }
  else
  {
    if ( v4 )
      goto LABEL_12;
    if ( *(char *)(a1 + 173) >= 0 )
      goto LABEL_46;
  }
  if ( (unsigned int)sub_7E3130(*(_QWORD *)(a1 + 120)) )
  {
LABEL_46:
    *(_BYTE *)(a1 + 177) = 3;
    goto LABEL_12;
  }
LABEL_32:
  if ( (*(_BYTE *)(a1 + 172) & 4) == 0 || *(_BYTE *)(a1 + 136) != 1 )
  {
LABEL_12:
    if ( (*(_BYTE *)(a1 + 88) & 0x70) != 0x10 )
      goto LABEL_13;
    goto LABEL_37;
  }
  if ( *(_BYTE *)(a1 + 177) != 1 )
    *(_QWORD *)(a1 + 184) = 0;
  *(_BYTE *)(a1 + 177) = 0;
  if ( (*(_BYTE *)(a1 + 88) & 0x70) == 0x10 )
  {
LABEL_37:
    v8 = *(const char **)(a1 + 8);
    if ( v8 && *v8 == 95 && !strcmp(v8, "__link") )
    {
      sub_7604D0(a1, 7u);
      *(_BYTE *)(a1 + 88) |= 4u;
    }
  }
LABEL_13:
  if ( qword_4F04C50 && a1 == *(_QWORD *)(qword_4F04C50 + 72LL) )
  {
    *(_BYTE *)(a1 + 88) &= ~4u;
    *(_QWORD *)(a1 + 170) &= 0xFFFFFFFFFFFFF7uLL;
  }
  v5 = dword_4F189E8;
  if ( (*(_BYTE *)(a1 + 156) & 3) == 1 )
    dword_4F189E8 = 1;
  sub_7EC360(a1, (__m128i *)(a1 + 177), (__int64 *)(a1 + 184));
  dword_4F189E8 = v5;
  result = *(_BYTE *)(a1 + 170) & 0x90;
  if ( (*(_BYTE *)(a1 + 170) & 0x90) == 0x10 && !*(_BYTE *)(a1 + 136) )
    result = sub_7E4C10(a1);
  if ( (*(_BYTE *)(a1 + 172) & 0x20) != 0 && *(_BYTE *)(a1 + 136) != 1 )
  {
    result = *(_BYTE *)(a1 + 88) & 0x70;
    if ( (_BYTE)result != 16 )
    {
      result = dword_4D0481C;
      if ( dword_4D0481C )
        return sub_7E4C10(a1);
    }
  }
  return result;
}
