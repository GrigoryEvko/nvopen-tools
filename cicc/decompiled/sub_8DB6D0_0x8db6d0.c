// Function: sub_8DB6D0
// Address: 0x8db6d0
//
__int64 __fastcall sub_8DB6D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r13
  __int64 v6; // r12
  __int64 v7; // rax
  char v8; // al
  __int64 v9; // r14
  __int64 v10; // r12
  const __m128i *v11; // rbx
  int v12; // r15d
  __m128i *v13; // rax

  v5 = a1;
  if ( a1 == a2 )
    return v5;
  v6 = a2;
  if ( a1 )
  {
    if ( a2 )
    {
      if ( dword_4F07588 )
      {
        v7 = *(_QWORD *)(a1 + 32);
        if ( *(_QWORD *)(a2 + 32) == v7 )
        {
          if ( v7 )
            return v5;
        }
      }
    }
  }
  v8 = *(_BYTE *)(a2 + 140);
  if ( *(_BYTE *)(a1 + 140) != 12 )
  {
    if ( v8 != 12 )
      goto LABEL_12;
    goto LABEL_10;
  }
  do
    v5 = *(_QWORD *)(v5 + 160);
  while ( *(_BYTE *)(v5 + 140) == 12 );
  if ( v8 == 12 )
  {
    do
LABEL_10:
      v6 = *(_QWORD *)(v6 + 160);
    while ( *(_BYTE *)(v6 + 140) == 12 );
  }
  if ( v5 == v6 )
    return v6;
LABEL_12:
  if ( (unsigned int)sub_8D97D0(v5, v6, 0, a4, a5) )
    return v5;
  if ( *(_BYTE *)(v5 + 140) != 6 )
    return 0;
  if ( (*(_BYTE *)(v5 + 168) & 1) != 0 )
    return 0;
  if ( *(_BYTE *)(v6 + 140) != 6 )
    return 0;
  if ( (*(_BYTE *)(v6 + 168) & 1) != 0 )
    return 0;
  v9 = sub_8D46C0(v5);
  v10 = sub_8D46C0(v6);
  v11 = (const __m128i *)sub_8DB6D0(v9, v10);
  if ( !v11 )
    return 0;
  v12 = 0;
  if ( (*(_BYTE *)(v9 + 140) & 0xFB) == 8 )
    v12 = sub_8D4C10(v9, dword_4F077C4 != 2);
  if ( (*(_BYTE *)(v10 + 140) & 0xFB) == 8 )
    v12 |= sub_8D4C10(v10, dword_4F077C4 != 2);
  v13 = sub_73C570(v11, v12);
  return sub_72D740(v13);
}
