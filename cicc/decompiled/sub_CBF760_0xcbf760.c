// Function: sub_CBF760
// Address: 0xcbf760
//
__int64 __fastcall sub_CBF760(_QWORD *a1, unsigned __int64 a2)
{
  unsigned __int64 v2; // rax
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rcx
  unsigned __int64 v5; // rax
  __int64 result; // rax
  unsigned __int128 v7; // rax
  __int64 v8; // rcx
  unsigned __int128 v9; // rax
  __int64 v10; // r8
  unsigned __int128 v11; // rax
  unsigned __int128 v12; // rax
  unsigned __int128 v13; // rax
  unsigned __int128 v14; // rax
  unsigned __int128 v15; // rax
  unsigned __int128 v16; // rax
  unsigned __int64 v17; // rax
  unsigned __int64 v18; // rax
  unsigned __int64 v19; // rax

  if ( a2 > 0x10 )
  {
    if ( a2 > 0x80 )
    {
      if ( a2 > 0xF0 )
        return sub_CBF100((__int64)a1, a2);
      else
        return sub_CBF370((__int64)a1, a2);
    }
    else
    {
      v7 = (a1[1] ^ 0x1CAD21F72C81017CuLL) * (unsigned __int128)(*a1 ^ 0xBE4BA423396CFEB8LL);
      v8 = (*((_QWORD *)&v7 + 1) ^ v7) - 0x61C8864E7A143579LL * a2;
      v9 = (*(_QWORD *)((char *)a1 + a2 - 8) ^ 0x1F67B3B7A4A44072uLL)
         * (unsigned __int128)(*(_QWORD *)((char *)a1 + a2 - 16) ^ 0xDB979083E96DD4DELL);
      v10 = v9 ^ *((_QWORD *)&v9 + 1);
      if ( a2 > 0x20 )
      {
        v11 = (a1[3] ^ 0x2172FFCC7DD05A82uLL) * (unsigned __int128)(a1[2] ^ 0x78E5C0CC4EE679CBuLL);
        v8 += *((_QWORD *)&v11 + 1) ^ v11;
        v12 = (*(_QWORD *)((char *)a1 + a2 - 24) ^ 0x4C263A81E69035E0uLL)
            * (unsigned __int128)(*(_QWORD *)((char *)a1 + a2 - 32) ^ 0x8E2443F7744608B8LL);
        v10 += *((_QWORD *)&v12 + 1) ^ v12;
        if ( a2 > 0x40 )
        {
          v13 = (a1[5] ^ 0xA32E531B8B65D088LL) * (unsigned __int128)(a1[4] ^ 0xCB00C391BB52283CLL);
          v8 += *((_QWORD *)&v13 + 1) ^ v13;
          v14 = (*(_QWORD *)((char *)a1 + a2 - 40) ^ 0xD8ACDEA946EF1938LL)
              * (unsigned __int128)(*(_QWORD *)((char *)a1 + a2 - 48) ^ 0x4EF90DA297486471uLL);
          v10 += *((_QWORD *)&v14 + 1) ^ v14;
          if ( a2 > 0x60 )
          {
            v15 = (a1[7] ^ 0x1D4F0BC7C7BBDCF9uLL) * (unsigned __int128)(a1[6] ^ 0x3F349CE33F76FAA8uLL);
            v8 += *((_QWORD *)&v15 + 1) ^ v15;
            v16 = (*(_QWORD *)((char *)a1 + a2 - 56) ^ 0x647378D9C97E9FC8uLL)
                * (unsigned __int128)(*(_QWORD *)((char *)a1 + a2 - 64) ^ 0x3159B4CD4BE0518AuLL);
            v10 += *((_QWORD *)&v16 + 1) ^ v16;
          }
        }
      }
      return ((0x165667919E3779F9LL * ((v8 + v10) ^ ((unsigned __int64)(v8 + v10) >> 37))) >> 32)
           ^ (0x165667919E3779F9LL * ((v8 + v10) ^ ((unsigned __int64)(v8 + v10) >> 37)));
    }
  }
  else if ( a2 <= 8 )
  {
    if ( a2 <= 3 )
    {
      result = 0x2D06800538D394C2LL;
      if ( a2 )
      {
        v19 = 0xC2B2AE3D27D4EB4FLL
            * (((*((unsigned __int8 *)a1 + (a2 >> 1)) << 24)
              | *((unsigned __int8 *)a1 + a2 - 1)
              | ((_DWORD)a2 << 8)
              | (*(unsigned __int8 *)a1 << 16))
             ^ 0x87275A9B);
        return ((0x165667B19E3779F9LL * ((v19 >> 29) ^ v19)) >> 32) ^ (0x165667B19E3779F9LL * ((v19 >> 29) ^ v19));
      }
    }
    else
    {
      v17 = (((unsigned __int64)*(unsigned int *)a1 << 32) | *(unsigned int *)((char *)a1 + a2 - 4))
          ^ 0xC73AB174C5ECD5A2LL;
      v18 = 0x9FB21C651E98DF25LL * (__ROL8__(v17, 24) ^ __ROR8__(v17, 15) ^ v17);
      return (0x9FB21C651E98DF25LL * (((v18 >> 35) + a2) ^ v18))
           ^ ((0x9FB21C651E98DF25LL * (((v18 >> 35) + a2) ^ v18)) >> 28);
    }
  }
  else
  {
    v2 = *a1 ^ 0x6782737BEA4239B9LL;
    v3 = *(_QWORD *)((char *)a1 + a2 - 8) ^ 0xAF56BC3B0996523ALL;
    v4 = _byteswap_uint64(v2) + a2 + v3;
    v5 = 0x165667919E3779F9LL
       * ((v4 + (((v3 * (unsigned __int128)v2) >> 64) ^ (v3 * v2)))
        ^ ((v4 + (((v3 * (unsigned __int128)v2) >> 64) ^ (v3 * v2))) >> 37));
    return HIDWORD(v5) ^ v5;
  }
  return result;
}
