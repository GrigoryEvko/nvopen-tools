// Function: sub_8F06E0
// Address: 0x8f06e0
//
void *__fastcall sub_8F06E0(_DWORD *a1, int a2)
{
  int v3; // r10d
  int v4; // esi
  int v5; // esi
  __int64 v6; // rdx
  int v7; // eax
  int v8; // eax
  int v9; // edx
  int v10; // eax
  __int128 v12; // [rsp+0h] [rbp-10h] BYREF

  v3 = a1[7];
  v12 = 0;
  v4 = v3 + 14;
  if ( v3 + 7 >= 0 )
    v4 = v3 + 7;
  v5 = v4 >> 3;
  if ( v3 <= 0 )
  {
    LOBYTE(v7) = 0;
    v9 = 0;
  }
  else
  {
    v6 = 0;
    v7 = 0;
    do
    {
      v8 = a2 * *((unsigned __int8 *)a1 + v6 + 12) + v7;
      *((_BYTE *)&v12 + v6++) = v8;
      v7 = v8 >> 8;
    }
    while ( v5 > (int)v6 );
    v9 = 1;
    if ( v5 > 0 )
      v9 = v5;
  }
  *((_BYTE *)&v12 + v9) = v7;
  v10 = a1[2] + 8;
  if ( (v3 & 7) != 0 )
    v10 += ((unsigned int)(v3 >> 31) >> 29) - ((((unsigned int)(v3 >> 31) >> 29) + (_BYTE)v3) & 7) + 8;
  a1[2] = v10;
  return sub_8EF4C0(a1, (char *)&v12, 8 * v5 + 8);
}
