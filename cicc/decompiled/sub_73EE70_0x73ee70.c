// Function: sub_73EE70
// Address: 0x73ee70
//
__int64 __fastcall sub_73EE70(__int64 *a1, __m128i **a2)
{
  __int64 v2; // r12
  __m128i *v3; // r13
  char v4; // al
  int i; // r14d
  __int64 result; // rax
  const __m128i *j; // rdi
  __int64 v8; // rdx
  __m128i *v9; // rax
  __int64 v10; // rax

  v2 = *a1;
  v3 = *a2;
  if ( !(unsigned int)sub_8D3D40(*a1) )
  {
    v4 = *(_BYTE *)(v2 + 140);
    if ( v4 != 12 )
    {
      i = 0;
      if ( (unsigned __int8)(v4 - 9) <= 2u )
        goto LABEL_6;
      if ( v4 != 8 )
        goto LABEL_5;
      goto LABEL_19;
    }
    if ( (*(_BYTE *)(v2 + 186) & 8) == 0 )
    {
LABEL_19:
      for ( i = sub_8D4C10(v2, dword_4F077C4 != 2); *(_BYTE *)(v2 + 140) == 12; v2 = *(_QWORD *)(v2 + 160) )
        ;
LABEL_5:
      *a1 = v2;
      goto LABEL_6;
    }
  }
  i = 0;
  v10 = sub_7CFE40(v2);
  *a1 = v10;
  v2 = v10;
LABEL_6:
  result = sub_8D2310(v3);
  if ( !(_DWORD)result )
    return result;
  for ( j = v3; j[8].m128i_i8[12] == 12; j = (const __m128i *)j[10].m128i_i64[0] )
    ;
  v8 = j[10].m128i_i64[1];
  if ( !*(_QWORD *)(v8 + 40) )
  {
    if ( !j[10].m128i_i64[0] )
      goto LABEL_27;
    goto LABEL_12;
  }
  if ( (i & ~(*(_BYTE *)(v8 + 18) & 0x7F)) != 0 )
  {
    if ( !j[10].m128i_i64[0] )
    {
LABEL_29:
      *(_BYTE *)(v8 + 18) |= i & 0x7F;
      goto LABEL_14;
    }
LABEL_12:
    v9 = sub_73EDA0(j, 1);
    v8 = v9[10].m128i_i64[1];
    v3 = v9;
    if ( *(_QWORD *)(v8 + 40) )
      goto LABEL_13;
LABEL_27:
    while ( *(_BYTE *)(v2 + 140) == 12 )
      v2 = *(_QWORD *)(v2 + 160);
    *(_BYTE *)(v8 + 21) |= 1u;
    *(_QWORD *)(v8 + 40) = v2;
    if ( !i )
      goto LABEL_14;
    goto LABEL_29;
  }
  result = *(_BYTE *)(v8 + 17) & 0x70;
  if ( (_BYTE)result != 48 )
    return result;
  if ( j[10].m128i_i64[0] )
    goto LABEL_12;
LABEL_13:
  if ( i )
    goto LABEL_29;
LABEL_14:
  result = *(unsigned __int8 *)(v8 + 17);
  if ( (*(_BYTE *)(v8 + 17) & 0x70) == 0x30 )
  {
    result = result & 0xFFFFFF8F | 0x20;
    *(_BYTE *)(v8 + 17) = result;
  }
  *a2 = v3;
  return result;
}
