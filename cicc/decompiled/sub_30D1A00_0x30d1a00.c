// Function: sub_30D1A00
// Address: 0x30d1a00
//
unsigned __int64 __fastcall sub_30D1A00(int a1)
{
  unsigned int v1; // edi

  v1 = 4 * a1 / 3u;
  return ((((((((((v1 + 1) | ((unsigned __int64)(v1 + 1) >> 1)) >> 2) | (v1 + 1) | ((unsigned __int64)(v1 + 1) >> 1)) >> 4)
            | (((v1 + 1) | ((unsigned __int64)(v1 + 1) >> 1)) >> 2)
            | (v1 + 1)
            | ((unsigned __int64)(v1 + 1) >> 1)) >> 8)
          | (((((v1 + 1) | ((unsigned __int64)(v1 + 1) >> 1)) >> 2) | (v1 + 1) | ((unsigned __int64)(v1 + 1) >> 1)) >> 4)
          | (((v1 + 1) | ((unsigned __int64)(v1 + 1) >> 1)) >> 2)
          | (v1 + 1)
          | ((unsigned __int64)(v1 + 1) >> 1)) >> 16)
        | (((((((v1 + 1) | ((unsigned __int64)(v1 + 1) >> 1)) >> 2) | (v1 + 1) | ((unsigned __int64)(v1 + 1) >> 1)) >> 4)
          | (((v1 + 1) | ((unsigned __int64)(v1 + 1) >> 1)) >> 2)
          | (v1 + 1)
          | ((unsigned __int64)(v1 + 1) >> 1)) >> 8)
        | (((((v1 + 1) | ((unsigned __int64)(v1 + 1) >> 1)) >> 2) | (v1 + 1) | ((unsigned __int64)(v1 + 1) >> 1)) >> 4)
        | (((v1 + 1) | ((unsigned __int64)(v1 + 1) >> 1)) >> 2)
        | (v1 + 1)
        | ((unsigned __int64)(v1 + 1) >> 1))
       + 1;
}
