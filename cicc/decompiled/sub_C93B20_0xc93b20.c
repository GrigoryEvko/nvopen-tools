// Function: sub_C93B20
// Address: 0xc93b20
//
__int64 __fastcall sub_C93B20(__int64 *a1, unsigned int a2, unsigned __int64 *a3)
{
  unsigned int v3; // r8d
  __int64 v5; // rcx
  char *v6; // rsi
  unsigned int v7; // edx
  unsigned __int64 v8; // r10
  unsigned __int64 v9; // r9
  char v10; // al

  v3 = a2;
  if ( !a2 )
    v3 = sub_C92F70(a1);
  v5 = a1[1];
  if ( !v5 )
    return 1;
  v6 = (char *)*a1;
  *a3 = 0;
  do
  {
    v10 = *v6;
    if ( *v6 <= 47 )
      goto LABEL_5;
    if ( v10 <= 57 )
    {
      v7 = (char)(v10 - 48);
      goto LABEL_7;
    }
    if ( v10 <= 96 )
    {
LABEL_5:
      if ( (unsigned __int8)(v10 - 65) > 0x19u )
        break;
      v7 = (char)(v10 - 55);
    }
    else
    {
      if ( v10 > 122 )
        break;
      v7 = (char)(v10 - 87);
    }
LABEL_7:
    if ( v3 <= v7 )
      break;
    v8 = *a3;
    v9 = *a3 * v3;
    *a3 = v9 + v7;
    if ( (v9 + v7) / v3 < v8 )
      return 1;
    ++v6;
    --v5;
  }
  while ( v5 );
  if ( a1[1] != v5 )
  {
    *a1 = (__int64)v6;
    a1[1] = v5;
    return 0;
  }
  return 1;
}
