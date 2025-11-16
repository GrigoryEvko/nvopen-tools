// Function: sub_28A9CB0
// Address: 0x28a9cb0
//
bool __fastcall sub_28A9CB0(
        _QWORD **a1,
        __int64 a2,
        __int64 a3,
        _QWORD *a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        __int128 a8,
        __int128 a9)
{
  __int64 v9; // r14
  __int64 v10; // rbx
  __int64 v12; // r15
  __int64 v13; // r13
  __m128i v14; // xmm0
  __m128i v15; // xmm1
  __m128i v16; // xmm2
  _QWORD *v17; // rdi
  bool result; // al
  __int64 v19; // rax
  __m128i v21[3]; // [rsp+10h] [rbp-70h] BYREF
  char v22; // [rsp+40h] [rbp-40h]

  v9 = a3 + 32;
  v10 = *(_QWORD *)(a2 + 40);
  if ( a3 + 32 == v10 )
    return 0;
  v12 = (__int64)(a1 + 1);
  while ( 1 )
  {
    if ( !v10 )
      BUG();
    v13 = *(_QWORD *)(v10 + 40);
    v14 = _mm_loadu_si128((const __m128i *)&a7);
    v15 = _mm_loadu_si128((const __m128i *)&a8);
    v16 = _mm_loadu_si128((const __m128i *)&a9);
    v22 = 1;
    v17 = *a1;
    v21[0] = v14;
    v21[1] = v15;
    v21[2] = v16;
    if ( (unsigned __int8)sub_CF63E0(v17, (unsigned __int8 *)v13, v21, v12) )
      break;
LABEL_15:
    v10 = *(_QWORD *)(v10 + 8);
    if ( v9 == v10 )
      return 0;
  }
  if ( *(_BYTE *)v13 == 85 )
  {
    v19 = *(_QWORD *)(v13 - 32);
    if ( v19 )
    {
      if ( !*(_BYTE *)v19 && *(_QWORD *)(v19 + 24) == *(_QWORD *)(v13 + 80) && (*(_BYTE *)(v19 + 33) & 0x20) != 0 )
      {
        result = a4 != 0 && *(_DWORD *)(v19 + 36) == 211;
        if ( result )
        {
          if ( *a4 )
            return result;
          *a4 = v13;
          goto LABEL_15;
        }
      }
    }
  }
  return 1;
}
