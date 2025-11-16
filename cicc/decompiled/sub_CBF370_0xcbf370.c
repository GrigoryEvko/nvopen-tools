// Function: sub_CBF370
// Address: 0xcbf370
//
unsigned __int64 __fastcall sub_CBF370(__int64 a1, unsigned __int64 a2)
{
  __int64 *v3; // rcx
  _QWORD *v4; // r10
  __int64 v5; // rdx
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // r11
  unsigned __int64 v8; // rsi
  unsigned __int64 v9; // rax
  __int64 v10; // rdx
  unsigned __int64 v11; // rsi
  unsigned int v12; // r10d
  int v13; // ecx
  int v14; // edx
  __int64 v15; // rax
  unsigned __int128 v16; // rax
  unsigned __int128 v17; // rax

  v3 = (__int64 *)&unk_3F6B458;
  v4 = (_QWORD *)(a1 + 8);
  v5 = 0x1CAD21F72C81017CLL;
  v6 = 0xBE4BA423396CFEB8LL;
  v7 = a2 >> 4;
  v8 = 0x9E3779B185EBCA87LL * a2;
  while ( 1 )
  {
    v9 = *(v4 - 1) ^ v6;
    v10 = *v4 ^ v5;
    v4 += 2;
    v8 += (((unsigned __int64)v10 * (unsigned __int128)v9) >> 64) ^ (v10 * v9);
    if ( &unk_3F6B4C8 == (_UNKNOWN *)v3 )
      break;
    v6 = *(v3 - 1);
    v5 = *v3;
    v3 += 2;
  }
  v11 = ((0x165667919E3779F9LL * ((v8 >> 37) ^ v8)) >> 32) ^ (0x165667919E3779F9LL * ((v8 >> 37) ^ v8));
  if ( (unsigned int)v7 > 8 )
  {
    v12 = 0;
    v13 = 8;
    do
    {
      v14 = v13;
      v15 = v12;
      ++v13;
      v12 += 16;
      v16 = (unsigned __int64)(*(_QWORD *)(a1 + (unsigned int)(16 * v14)) ^ *(_QWORD *)((char *)&unk_3F6B440 + v15 + 3))
          * (unsigned __int128)(unsigned __int64)(*(_QWORD *)(a1 + (unsigned int)(16 * v14) + 8)
                                                ^ *(_QWORD *)((char *)&unk_3F6B440 + v15 + 11));
      v11 += *((_QWORD *)&v16 + 1) ^ v16;
    }
    while ( (_DWORD)v7 != v13 );
  }
  v17 = (*(_QWORD *)(a1 + a2 - 16 + 8) ^ 0xEBD33483ACC5EA64LL)
      * (unsigned __int128)(*(_QWORD *)(a1 + a2 - 16) ^ 0x7378D9C97E9FC831uLL);
  *(_QWORD *)&v17 = 0x165667919E3779F9LL
                  * (((v17 ^ *((_QWORD *)&v17 + 1)) + v11)
                   ^ ((((unsigned __int64)v17 ^ *((_QWORD *)&v17 + 1)) + v11) >> 37));
  return DWORD1(v17) ^ v17;
}
