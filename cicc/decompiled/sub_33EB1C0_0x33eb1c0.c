// Function: sub_33EB1C0
// Address: 0x33eb1c0
//
__m128i *__fastcall sub_33EB1C0(
        __int64 a1,
        __int32 a2,
        __int64 a3,
        unsigned __int64 a4,
        __int32 a5,
        int a6,
        unsigned __int64 *a7,
        __int64 a8,
        __int64 a9,
        __int64 a10,
        __int128 a11,
        __int64 a12,
        unsigned __int16 a13,
        __int64 a14,
        __int64 a15)
{
  int v16; // ecx
  unsigned __int16 v21; // si
  int v22; // edx
  const __m128i *v23; // rax
  __int64 v25; // rax
  __int64 v26; // rdi
  char v27; // dl
  char v28; // al
  __int64 v29; // rdx
  unsigned __int64 v30; // rdi
  unsigned __int16 v31; // [rsp+8h] [rbp-48h]
  unsigned __int8 v32; // [rsp+Fh] [rbp-41h]

  v16 = a6;
  v21 = a13;
  v22 = a14;
  if ( a14 != -1 && a14 != 0xBFFFFFFFFFFFFFFELL && (a14 & 0x3FFFFFFFFFFFFFFFLL) == 0 )
  {
    if ( (_WORD)a9 )
    {
      if ( (_WORD)a9 == 1 || (unsigned __int16)(a9 - 504) <= 7u )
        BUG();
      v29 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)a9 - 16];
      v28 = byte_444C4A0[16 * (unsigned __int16)a9 - 8];
    }
    else
    {
      v31 = a13;
      v32 = a6;
      v25 = sub_3007260((__int64)&a9);
      v16 = v32;
      v21 = v31;
      v26 = v25;
      v28 = v27;
      v29 = v26;
    }
    v30 = (unsigned __int64)(v29 + 7) >> 3;
    v22 = v30;
    if ( !v28 )
      v22 = v30;
  }
  v23 = (const __m128i *)sub_2E7BD70(*(_QWORD **)(a1 + 40), v21, v22, v16, a15, 0, a11, a12, 1u, 0, 0);
  return sub_33EA9D0((_QWORD *)a1, a2, a3, a4, a5, v23, a7, a8, a9, a10);
}
