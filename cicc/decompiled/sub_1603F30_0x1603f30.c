// Function: sub_1603F30
// Address: 0x1603f30
//
unsigned __int64 __fastcall sub_1603F30(int a1)
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
