// Function: sub_CBF100
// Address: 0xcbf100
//
unsigned __int64 __fastcall sub_CBF100(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  unsigned __int64 *v3; // r9
  __int64 v4; // r10
  unsigned __int64 i; // rdx
  __int64 *v6; // rcx
  __int64 j; // rax
  __int64 v8; // rsi
  char *v9; // rdx
  __int64 *v10; // rsi
  unsigned __int64 v11; // rcx
  __int64 v12; // rax
  unsigned __int64 v13; // r9
  unsigned __int64 v14; // rax
  unsigned __int64 *v15; // r8
  unsigned __int64 v16; // rdx
  unsigned __int64 *v17; // r10
  __int64 *v18; // rsi
  __int64 k; // rax
  __int64 v20; // rbx
  __int64 *v21; // rsi
  __int64 m; // rax
  __int64 v23; // r8
  unsigned __int64 v24; // rdx
  __int64 v26; // [rsp+8h] [rbp-88h]
  unsigned __int64 v28; // [rsp+18h] [rbp-78h]
  _QWORD v29[8]; // [rsp+20h] [rbp-70h] BYREF
  char v30; // [rsp+60h] [rbp-30h] BYREF

  v29[0] = 3266489917LL;
  v29[1] = 0x9E3779B185EBCA87LL;
  v29[2] = 0xC2B2AE3D27D4EB4FLL;
  v29[3] = 0x165667B19E3779F9LL;
  v29[4] = 0x85EBCA77C2B2AE63LL;
  v29[5] = 2246822519LL;
  v26 = a2 - 1;
  v29[6] = 0x27D4EB2F165667C5LL;
  v29[7] = 2654435761LL;
  if ( (unsigned __int64)(a2 - 1) >> 10 )
  {
    v2 = 0;
    v28 = (unsigned __int64)(a2 - 1) >> 10 << 10;
    do
    {
      v3 = (unsigned __int64 *)&unk_3F6B440;
      v4 = v2;
      for ( i = 0xBE4BA423396CFEB8LL; ; i = *v3 )
      {
        v6 = (__int64 *)(a1 + v4);
        for ( j = 0; ; i = v3[j] )
        {
          v8 = *v6++;
          v29[j ^ 1] += v8;
          v29[j++] += ((unsigned int)v8 ^ (unsigned int)i) * ((v8 ^ i) >> 32);
          if ( j == 8 )
            break;
        }
        ++v3;
        v4 += 64;
        if ( &unk_3F6B4C0 == (_UNKNOWN *)v3 )
          break;
      }
      v9 = (char *)v29;
      v10 = (__int64 *)&unk_3F6B4C0;
      do
      {
        v11 = *(_QWORD *)v9;
        v12 = *v10;
        v9 += 8;
        ++v10;
        *((_QWORD *)v9 - 1) = 2654435761u * ((v11 >> 47) ^ v11 ^ v12);
      }
      while ( &v30 != v9 );
      v2 += 1024;
    }
    while ( v28 != v2 );
  }
  v13 = v26 & 0xFFFFFFFFFFFFFC00LL;
  v14 = (unsigned __int64)(v26 & 0x3FF) >> 6;
  if ( v14 )
  {
    v15 = (unsigned __int64 *)&unk_3F6B440;
    v16 = 0xBE4BA423396CFEB8LL;
    v17 = (unsigned __int64 *)((char *)&unk_3F6B440 + 8 * v14);
    while ( 1 )
    {
      v18 = (__int64 *)(a1 + v13);
      for ( k = 0; ; v16 = v15[k] )
      {
        v20 = *v18++;
        v29[k ^ 1] += v20;
        v29[k++] += ((unsigned int)v20 ^ (unsigned int)v16) * ((v20 ^ v16) >> 32);
        if ( k == 8 )
          break;
      }
      ++v15;
      v13 += 64LL;
      if ( v17 == v15 )
        break;
      v16 = *v15;
    }
  }
  v21 = (__int64 *)(a1 + a2 - 64);
  for ( m = 0; m != 8; ++m )
  {
    v23 = *v21++;
    v24 = v23 ^ *((_QWORD *)&unk_3F6B4B9 + m);
    v29[m ^ 1] += v23;
    v29[m] += (unsigned int)v24 * HIDWORD(v24);
  }
  return sub_CBF0B0((__int64)v29, (__int64)&unk_3F6B44B, 0x9E3779B185EBCA87LL * a2);
}
