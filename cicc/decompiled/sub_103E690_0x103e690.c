// Function: sub_103E690
// Address: 0x103e690
//
char __fastcall sub_103E690(__int64 a1)
{
  _BYTE *v1; // rax
  char v2; // dl
  __int64 v3; // rax
  __m128i v4; // xmm0
  __m128i v5; // xmm1
  __m128i v6; // xmm2
  bool v7; // zf
  __int64 v8; // rax
  __int64 v9; // r8
  _BYTE *v10; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  unsigned __int8 *v13; // rsi
  __int64 *v14; // rcx
  __int64 *v15; // rax
  _QWORD v17[4]; // [rsp+0h] [rbp-60h] BYREF
  _QWORD *v18; // [rsp+20h] [rbp-40h]
  __int64 v19; // [rsp+28h] [rbp-38h]
  _QWORD v20[6]; // [rsp+30h] [rbp-30h] BYREF

  v1 = *(_BYTE **)(a1 + 56);
  v2 = *v1;
  if ( *v1 == 28 )
  {
    v3 = *(_QWORD *)(*((_QWORD *)v1 - 1) + 32LL * *(unsigned int *)(a1 + 64));
  }
  else
  {
    v14 = (__int64 *)(v1 - 32);
    v15 = (__int64 *)(v1 - 64);
    if ( v2 == 26 )
      v15 = v14;
    v3 = *v15;
  }
  v4 = _mm_loadu_si128((const __m128i *)(a1 + 72));
  v5 = _mm_loadu_si128((const __m128i *)(a1 + 88));
  *(_QWORD *)a1 = v3;
  v6 = _mm_loadu_si128((const __m128i *)(a1 + 104));
  v7 = *(_BYTE *)(a1 + 136) == 0;
  *(__m128i *)(a1 + 8) = v4;
  *(__m128i *)(a1 + 24) = v5;
  *(__m128i *)(a1 + 40) = v6;
  if ( !v7 && *(_QWORD *)(a1 + 72) )
  {
    v8 = sub_AA4E30(*(_QWORD *)(*(_QWORD *)(a1 + 120) + 64LL));
    v17[2] = 0;
    v9 = v8;
    v10 = *(_BYTE **)(a1 + 72);
    v17[3] = 0;
    v17[1] = v9;
    v17[0] = v10;
    v18 = v20;
    v19 = 0x400000000LL;
    if ( *v10 > 0x1Cu )
    {
      v20[0] = v10;
      LODWORD(v19) = 1;
    }
    v11 = *(_QWORD *)(a1 + 56);
    if ( *(_BYTE *)v11 != 28 )
      BUG();
    v12 = sub_104B4A0(
            v17,
            *(_QWORD *)(*(_QWORD *)(a1 + 120) + 64LL),
            *(_QWORD *)(*(_QWORD *)(v11 - 8) + 32LL * *(unsigned int *)(v11 + 76) + 8LL * *(unsigned int *)(a1 + 64)),
            *(_QWORD *)(a1 + 128),
            1);
    v13 = (unsigned __int8 *)v12;
    if ( v12 )
    {
      if ( *(_QWORD *)(a1 + 8) != v12 )
        *(_QWORD *)(a1 + 8) = v12;
    }
    else
    {
      v13 = *(unsigned __int8 **)(a1 + 8);
    }
    LOBYTE(v3) = sub_103E5A0(a1, v13);
    if ( !(_BYTE)v3 )
      *(_QWORD *)(a1 + 16) = -1;
    if ( v18 != v20 )
      LOBYTE(v3) = _libc_free(v18, v13);
  }
  return v3;
}
