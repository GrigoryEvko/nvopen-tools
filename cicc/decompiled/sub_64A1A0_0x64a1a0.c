// Function: sub_64A1A0
// Address: 0x64a1a0
//
__int64 __fastcall sub_64A1A0(__int64 a1, __int64 a2)
{
  __int64 i; // r8
  char v3; // cl
  __int64 v4; // rbx
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rsi
  __int64 v10; // rdi
  __int64 v11; // rax

  for ( i = a2; *(_BYTE *)(a1 + 140) == 12; a1 = *(_QWORD *)(a1 + 160) )
    ;
  v3 = *(_BYTE *)(a2 + 140);
  v4 = *(_QWORD *)(a1 + 168);
  v5 = a2;
  if ( v3 == 12 )
  {
    do
      v5 = *(_QWORD *)(v5 + 160);
    while ( *(_BYTE *)(v5 + 140) == 12 );
  }
  v6 = *(_QWORD *)(v5 + 168);
  v7 = *(_QWORD *)(v4 + 40);
  v8 = *(_QWORD *)(v6 + 40);
  if ( v7 != v8 && (!v8 || !v7 || !dword_4F07588 || (v10 = *(_QWORD *)(v7 + 32), *(_QWORD *)(v8 + 32) != v10) || !v10)
    || ((*(_BYTE *)(v6 + 18) ^ *(_BYTE *)(v4 + 18)) & 0x7F) != 0
    || ((*(_WORD *)(v6 + 18) ^ *(_WORD *)(v4 + 18)) & 0x3F80) != 0
    || ((*(_BYTE *)(v6 + 17) ^ *(_BYTE *)(v4 + 17)) & 0x70) != 0 )
  {
    if ( v3 == 12 )
    {
      v11 = sub_73EDA0(i, 0);
      v7 = *(_QWORD *)(v4 + 40);
      i = v11;
      v6 = *(_QWORD *)(v11 + 168);
    }
    *(_QWORD *)(v6 + 40) = v7;
    *(_BYTE *)(v6 + 21) = *(_BYTE *)(v4 + 21) & 1 | *(_BYTE *)(v6 + 21) & 0xFE;
    *(_BYTE *)(v6 + 18) = *(_BYTE *)(v4 + 18) & 0x7F | *(_BYTE *)(v6 + 18) & 0x80;
    *(_WORD *)(v6 + 18) = *(_WORD *)(v4 + 18) & 0x3F80 | *(_WORD *)(v6 + 18) & 0xC07F;
    *(_BYTE *)(v6 + 17) = *(_BYTE *)(v4 + 17) & 0x70 | *(_BYTE *)(v6 + 17) & 0x8F;
  }
  return i;
}
