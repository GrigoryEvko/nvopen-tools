// Function: sub_39F8CF0
// Address: 0x39f8cf0
//
__int64 __fastcall sub_39F8CF0(__int64 a1)
{
  size_t v1; // rax
  unsigned __int8 v2; // cl
  char *v3; // rax
  unsigned int v4; // r8d
  char *v6; // rdx
  char *v7; // rbx
  char v8; // dl
  unsigned __int64 v9[4]; // [rsp+8h] [rbp-20h] BYREF

  v1 = strlen((const char *)(a1 + 9));
  v2 = *(_BYTE *)(a1 + 8);
  v3 = (char *)(a1 + 9 + v1 + 1);
  if ( v2 > 3u )
  {
    v4 = 255;
    if ( *v3 != 8 || v3[1] )
      return v4;
    v3 += 2;
  }
  if ( *(_BYTE *)(a1 + 9) != 122 )
    return 0;
  do
    ++v3;
  while ( *(v3 - 1) < 0 );
  do
    v6 = v3++;
  while ( *(v3 - 1) < 0 );
  if ( v2 == 1 )
  {
    v3 = v6 + 2;
  }
  else
  {
    do
      ++v3;
    while ( *(v3 - 1) < 0 );
  }
  v7 = (char *)(a1 + 10);
  do
    ++v3;
  while ( *(v3 - 1) < 0 );
  v8 = *(_BYTE *)(a1 + 10);
  if ( v8 != 82 )
  {
    while ( 1 )
    {
      while ( v8 == 80 )
      {
        ++v7;
        v3 = sub_39F8BA0(*v3 & 0x7F, 0, v3 + 1, v9);
        v8 = *v7;
        if ( *v7 == 82 )
          return (unsigned __int8)*v3;
      }
      if ( v8 != 76 && v8 != 66 )
        break;
      v8 = *++v7;
      ++v3;
      if ( v8 == 82 )
        return (unsigned __int8)*v3;
    }
    return 0;
  }
  return (unsigned __int8)*v3;
}
