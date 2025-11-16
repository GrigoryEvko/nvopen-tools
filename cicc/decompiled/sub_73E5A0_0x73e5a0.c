// Function: sub_73E5A0
// Address: 0x73e5a0
//
_BYTE *__fastcall sub_73E5A0(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // rdi
  char v4; // al
  __int64 v5; // rax
  char v6; // dl
  _BYTE *result; // rax
  int v8; // ebx
  __int64 v9; // rax
  _QWORD *v10; // r14
  __m128i *v11; // rax

  v2 = a1;
  if ( !*(_BYTE *)(a1 + 24) )
    return (_BYTE *)v2;
  v3 = *(_QWORD *)a1;
  v4 = *(_BYTE *)(v3 + 140);
  if ( v4 == 12 )
  {
    v5 = v3;
    do
    {
      v5 = *(_QWORD *)(v5 + 160);
      v6 = *(_BYTE *)(v5 + 140);
    }
    while ( v6 == 12 );
    if ( !v6 )
      return (_BYTE *)v2;
LABEL_7:
    v8 = sub_8D4C10(v3, dword_4F077C4 != 2);
    goto LABEL_8;
  }
  if ( !v4 )
    return (_BYTE *)v2;
  if ( (v4 & 0xFB) == 8 )
    goto LABEL_7;
  v8 = 0;
LABEL_8:
  v9 = *(_QWORD *)(a2 + 112);
  v10 = *(_QWORD **)(v9 + 16);
  if ( (*(_BYTE *)(a2 + 96) & 2) == 0 )
    v10 = *(_QWORD **)(v9 + 8);
  if ( !v10 )
    return (_BYTE *)v2;
  do
  {
    v11 = sub_73C570(*(const __m128i **)(v10[2] + 40LL), v8);
    result = sub_73DBF0(0xEu, (__int64)v11, v2);
    result[27] |= 2u;
    v10 = (_QWORD *)*v10;
    v2 = (__int64)result;
  }
  while ( v10 );
  return result;
}
