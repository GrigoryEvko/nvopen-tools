// Function: sub_14CE540
// Address: 0x14ce540
//
__int64 __fastcall sub_14CE540(__int64 a1)
{
  void (*v1)(void); // rax
  int v2; // r12d
  __int64 result; // rax
  __int64 v4; // r13
  unsigned int v5; // r12d
  _QWORD *v6; // r12
  unsigned int v7; // r13d
  unsigned __int64 v8; // rax
  __int64 v9; // rdx
  __int64 i; // rdx
  char v11; // cl
  _QWORD *v12; // rbx
  char v13; // al
  __int64 v14; // rax
  bool v15; // zf
  _QWORD v16[2]; // [rsp+8h] [rbp-48h] BYREF
  __int64 v17; // [rsp+18h] [rbp-38h]
  __int64 v18; // [rsp+20h] [rbp-30h]

  v1 = *(void (**)(void))(*(_QWORD *)a1 + 128LL);
  if ( (char *)v1 == (char *)sub_14CE4E0 )
  {
    if ( byte_4F9D820 )
      sub_14CE140(a1);
  }
  else
  {
    v1();
  }
  v2 = *(_DWORD *)(a1 + 176);
  sub_14CB1D0(a1 + 160);
  if ( v2 )
  {
    v4 = 64;
    v5 = v2 - 1;
    if ( v5 )
    {
      _BitScanReverse(&v5, v5);
      v4 = (unsigned int)(1 << (33 - (v5 ^ 0x1F)));
      if ( (int)v4 < 64 )
        v4 = 64;
    }
    v6 = *(_QWORD **)(a1 + 168);
    if ( *(_DWORD *)(a1 + 184) == (_DWORD)v4 )
    {
      v16[0] = 2;
      *(_QWORD *)(a1 + 176) = 0;
      v16[1] = 0;
      v17 = -8;
      v12 = &v6[6 * v4];
      v18 = 0;
      do
      {
        if ( v6 )
        {
          v13 = v16[0];
          v6[2] = 0;
          v6[1] = v13 & 6;
          v14 = v17;
          v15 = v17 == 0;
          v6[3] = v17;
          if ( v14 != -8 && !v15 && v14 != -16 )
            sub_1649AC0(v6 + 1, v16[0] & 0xFFFFFFFFFFFFFFF8LL);
          *v6 = &unk_49ECBF8;
          v6[4] = v18;
        }
        v6 += 6;
      }
      while ( v12 != v6 );
      result = v17;
      if ( v17 != 0 && v17 != -8 && v17 != -16 )
        return sub_1649B30(v16);
    }
    else
    {
      j___libc_free_0(*(_QWORD *)(a1 + 168));
      v7 = 4 * (int)v4 / 3u;
      v8 = ((((((((((v7 + 1) | ((unsigned __int64)(v7 + 1) >> 1)) >> 2) | (v7 + 1) | ((unsigned __int64)(v7 + 1) >> 1)) >> 4)
              | (((v7 + 1) | ((unsigned __int64)(v7 + 1) >> 1)) >> 2)
              | (v7 + 1)
              | ((unsigned __int64)(v7 + 1) >> 1)) >> 8)
            | (((((v7 + 1) | ((unsigned __int64)(v7 + 1) >> 1)) >> 2) | (v7 + 1) | ((unsigned __int64)(v7 + 1) >> 1)) >> 4)
            | (((v7 + 1) | ((unsigned __int64)(v7 + 1) >> 1)) >> 2)
            | (v7 + 1)
            | ((unsigned __int64)(v7 + 1) >> 1)) >> 16)
          | (((((((v7 + 1) | ((unsigned __int64)(v7 + 1) >> 1)) >> 2) | (v7 + 1) | ((unsigned __int64)(v7 + 1) >> 1)) >> 4)
            | (((v7 + 1) | ((unsigned __int64)(v7 + 1) >> 1)) >> 2)
            | (v7 + 1)
            | ((unsigned __int64)(v7 + 1) >> 1)) >> 8)
          | (((((v7 + 1) | ((unsigned __int64)(v7 + 1) >> 1)) >> 2) | (v7 + 1) | ((unsigned __int64)(v7 + 1) >> 1)) >> 4)
          | (((v7 + 1) | ((unsigned __int64)(v7 + 1) >> 1)) >> 2)
          | (v7 + 1)
          | ((unsigned __int64)(v7 + 1) >> 1))
         + 1;
      *(_DWORD *)(a1 + 184) = v8;
      result = sub_22077B0(48 * v8);
      *(_QWORD *)(a1 + 176) = 0;
      *(_QWORD *)(a1 + 168) = result;
      v9 = *(unsigned int *)(a1 + 184);
      v16[0] = 2;
      v18 = 0;
      for ( i = result + 48 * v9; i != result; result += 48 )
      {
        if ( result )
        {
          v11 = v16[0];
          *(_QWORD *)(result + 16) = 0;
          *(_QWORD *)(result + 24) = -8;
          *(_QWORD *)result = &unk_49ECBF8;
          *(_QWORD *)(result + 8) = v11 & 6;
          *(_QWORD *)(result + 32) = v18;
        }
      }
    }
  }
  else
  {
    result = *(unsigned int *)(a1 + 184);
    if ( (_DWORD)result )
    {
      result = j___libc_free_0(*(_QWORD *)(a1 + 168));
      *(_QWORD *)(a1 + 168) = 0;
      *(_QWORD *)(a1 + 176) = 0;
      *(_DWORD *)(a1 + 184) = 0;
    }
    else
    {
      *(_QWORD *)(a1 + 176) = 0;
    }
  }
  return result;
}
