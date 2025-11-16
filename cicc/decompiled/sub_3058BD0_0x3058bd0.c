// Function: sub_3058BD0
// Address: 0x3058bd0
//
__int64 __fastcall sub_3058BD0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  void *v7; // rdi
  int v8; // r15d
  unsigned int v9; // ebx
  int i; // ebx
  unsigned int v11; // edx

  v7 = (void *)(a1 + 16);
  v8 = *(_DWORD *)(a2 + 16);
  *(_QWORD *)a1 = v7;
  *(_QWORD *)(a1 + 8) = 0x600000000LL;
  v9 = (unsigned int)(v8 + 63) >> 6;
  if ( v9 > 6 )
  {
    sub_C8D5F0(a1, v7, v9, 8u, a5, a6);
    memset(*(void **)a1, 0, 8LL * v9);
    *(_DWORD *)(a1 + 8) = v9;
  }
  else
  {
    if ( v9 )
      memset(v7, 0, (size_t)v7 + 8 * v9 - a1 - 16);
    *(_DWORD *)(a1 + 8) = v9;
  }
  *(_DWORD *)(a1 + 64) = v8;
  for ( i = 2; i != 34; ++i )
  {
    v11 = i;
    sub_2FF62B0(a2, (_QWORD *)a1, v11);
  }
  sub_2FF62B0(a2, (_QWORD *)a1, 0x4Fu);
  sub_2FF62B0(a2, (_QWORD *)a1, 0x51u);
  sub_2FF62B0(a2, (_QWORD *)a1, 0x50u);
  sub_2FF62B0(a2, (_QWORD *)a1, 0x52u);
  sub_2FF62B0(a2, (_QWORD *)a1, 1u);
  return a1;
}
