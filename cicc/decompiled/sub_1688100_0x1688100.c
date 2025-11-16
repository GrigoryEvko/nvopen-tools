// Function: sub_1688100
// Address: 0x1688100
//
__int64 __fastcall sub_1688100(__int64 a1)
{
  unsigned int v1; // r8d
  int v2; // esi
  int v3; // edx
  unsigned int v4; // esi
  int v6; // r9d
  int v7; // eax
  __int64 v8; // rdx

  v1 = 0;
  v2 = 0;
  if ( *(_BYTE *)a1 )
  {
    while ( *(_BYTE *)(a1 + 1) )
    {
      if ( !*(_BYTE *)(a1 + 2) || !*(_BYTE *)(a1 + 3) )
      {
        v6 = 2 - ((*(_BYTE *)(a1 + 2) == 0) - 1);
        goto LABEL_8;
      }
      v3 = -862048943 * *(_DWORD *)a1;
      a1 += 4;
      v1 += 4;
      v2 = 5 * __ROL4__(v2 ^ (461845907 * __ROL4__(v3, 15)), 13) - 430675100;
      if ( !*(_BYTE *)a1 )
        goto LABEL_6;
    }
    v6 = 1;
LABEL_8:
    v7 = 0;
    v8 = v6;
    do
    {
      --v8;
      v7 = *(unsigned __int8 *)(a1 + v8) | (v7 << 8);
    }
    while ( v8 );
    v1 += v6;
    v2 ^= 461845907 * __ROL4__(-862048943 * v7, 15);
  }
LABEL_6:
  v4 = -2048144789 * (v2 ^ v1 ^ ((v2 ^ v1) >> 16));
  return ((-1028477387 * (v4 ^ (v4 >> 13))) >> 16) ^ (-1028477387 * (v4 ^ (v4 >> 13)));
}
