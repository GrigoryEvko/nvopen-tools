// Function: sub_22551E0
// Address: 0x22551e0
//
char __fastcall sub_22551E0(__int64 a1, __int64 a2, unsigned __int8 *a3, __int64 a4)
{
  unsigned __int64 v4; // rcx
  unsigned __int64 v5; // r9
  unsigned __int64 v7; // rax
  unsigned __int8 v8; // di
  unsigned __int8 v9; // si
  bool v10; // r8
  char result; // al
  unsigned __int8 v12; // si
  unsigned __int8 v13; // al

  v4 = a4 - 1;
  v5 = a2 - 1;
  if ( a2 - 1 > v4 )
    v5 = v4;
  v7 = 0;
  if ( v5 )
  {
    do
    {
      v8 = a3[v4];
      v9 = *(_BYTE *)(a1 + v7);
      --v4;
      ++v7;
      v10 = v8 != v9;
    }
    while ( v7 < v5 && v8 == v9 );
    result = v8 == v9;
  }
  else
  {
    v10 = 0;
    result = 1;
  }
  v12 = *(_BYTE *)(a1 + v5);
  if ( v4 && !v10 )
  {
    do
      v13 = a3[v4--];
    while ( v4 && v13 == v12 );
    result = v13 == v12;
  }
  if ( (unsigned __int8)(v12 - 1) <= 0x7Du )
    return ((char)*a3 <= (char)v12) & result;
  return result;
}
