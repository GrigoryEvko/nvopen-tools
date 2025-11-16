// Function: sub_20C33D0
// Address: 0x20c33d0
//
__int64 __fastcall sub_20C33D0(__int64 a1, int *a2)
{
  __int64 v4; // rax
  unsigned int v5; // esi
  __m128i v6; // xmm0
  __int64 v7; // r8
  __int64 v8; // rdx
  __int64 v9; // r12
  unsigned int v10; // ecx
  __int64 v11; // rax
  _BOOL8 v13; // rdi

  v4 = sub_22077B0(56);
  v5 = *a2;
  v6 = _mm_loadu_si128((const __m128i *)(a2 + 2));
  v7 = a1 + 8;
  v8 = *(_QWORD *)(a1 + 16);
  v9 = v4;
  *(_DWORD *)(v4 + 32) = v5;
  *(__m128i *)(v4 + 40) = v6;
  if ( v8 )
  {
    while ( 1 )
    {
      v10 = *(_DWORD *)(v8 + 32);
      v11 = *(_QWORD *)(v8 + 24);
      if ( v10 > v5 )
        v11 = *(_QWORD *)(v8 + 16);
      if ( !v11 )
        break;
      v8 = v11;
    }
    v13 = 1;
    if ( v7 != v8 )
      v13 = v10 > v5;
  }
  else
  {
    v8 = a1 + 8;
    v13 = 1;
  }
  sub_220F040(v13, v9, v8, a1 + 8);
  ++*(_QWORD *)(a1 + 40);
  return v9;
}
