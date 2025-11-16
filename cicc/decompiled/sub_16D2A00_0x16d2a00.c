// Function: sub_16D2A00
// Address: 0x16d2a00
//
__int64 __fastcall sub_16D2A00(__int64 a1, unsigned int a2, unsigned __int64 *a3)
{
  __int64 v3; // r10
  unsigned __int64 *v4; // r9
  unsigned int v5; // r8d
  __int64 v6; // rcx
  char *v7; // rsi
  unsigned int v8; // edx
  unsigned __int64 v9; // rbx
  unsigned __int64 v10; // r11
  char v11; // al

  v3 = a1;
  v4 = a3;
  v5 = a2;
  if ( !a2 )
    v5 = sub_16D1EB0(a1);
  v6 = *(_QWORD *)(v3 + 8);
  if ( !v6 )
    return 1;
  v7 = *(char **)v3;
  *v4 = 0;
  do
  {
    v11 = *v7;
    if ( *v7 <= 47 )
      goto LABEL_5;
    if ( v11 <= 57 )
    {
      v8 = (char)(v11 - 48);
      goto LABEL_7;
    }
    if ( v11 <= 96 )
    {
LABEL_5:
      if ( (unsigned __int8)(v11 - 65) > 0x19u )
        break;
      v8 = (char)(v11 - 55);
    }
    else
    {
      if ( v11 > 122 )
        break;
      v8 = (char)(v11 - 87);
    }
LABEL_7:
    if ( v5 <= v8 )
      break;
    v9 = *v4;
    v10 = *v4 * v5;
    *v4 = v10 + v8;
    if ( (v10 + v8) / v5 < v9 )
      return 1;
    ++v7;
    --v6;
  }
  while ( v6 );
  if ( *(_QWORD *)(v3 + 8) != v6 )
  {
    *(_QWORD *)v3 = v7;
    *(_QWORD *)(v3 + 8) = v6;
    return 0;
  }
  return 1;
}
