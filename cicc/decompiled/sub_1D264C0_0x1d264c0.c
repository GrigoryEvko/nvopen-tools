// Function: sub_1D264C0
// Address: 0x1d264c0
//
__int64 __fastcall sub_1D264C0(
        _QWORD *a1,
        unsigned __int64 a2,
        __int64 a3,
        unsigned int a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        __int64 a8,
        __int64 a9,
        __int128 a10,
        __int128 a11,
        __int64 a12,
        __int64 a13,
        __int64 a14,
        int a15,
        unsigned int a16,
        __int64 a17,
        __int64 a18)
{
  __int64 v18; // r15
  char v19; // r14
  int v20; // r13d
  __int64 v22; // rcx
  int v23; // edx
  int v24; // r10d
  __int64 v25; // rax
  int v27; // eax
  int v28; // eax
  __int64 v29; // rax
  unsigned int v30; // edx
  __int64 *v31; // r8
  __int64 v32; // r8
  int v33; // [rsp+8h] [rbp-68h]
  unsigned __int16 v35; // [rsp+1Ch] [rbp-54h]
  unsigned int v36; // [rsp+1Ch] [rbp-54h]
  __m128i v37; // [rsp+20h] [rbp-50h] BYREF
  __int64 v38; // [rsp+30h] [rbp-40h]

  v18 = a6;
  v19 = a3;
  v20 = a2;
  v22 = a16;
  if ( !a15 )
  {
    a2 = (unsigned int)a13;
    v36 = a16;
    v28 = sub_1D172F0((__int64)a1, (unsigned int)a13, a14);
    v22 = v36;
    a15 = v28;
  }
  v35 = v22 | 1;
  if ( (a11 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
  {
    a3 = a10;
    a6 = a8;
    v22 = *(unsigned __int16 *)(a10 + 24);
    if ( (_WORD)v22 == 32 || (_DWORD)v22 == 10 )
    {
      v29 = *(_QWORD *)(a10 + 88);
      v30 = *(_DWORD *)(v29 + 32);
      v31 = *(__int64 **)(v29 + 24);
      if ( v30 > 0x40 )
        v32 = *v31;
      else
        v32 = (__int64)((_QWORD)v31 << (64 - (unsigned __int8)v30)) >> (64 - (unsigned __int8)v30);
      a2 = (unsigned __int64)&a11;
      sub_1D13370(&v37, (const __m128i *)&a11, (__int64)a1, a8, v32);
    }
    else if ( (_WORD)v22 == 48 )
    {
      a2 = (unsigned __int64)&a11;
      sub_1D13370(&v37, (const __m128i *)&a11, (__int64)a1, a8, 0);
    }
    else
    {
      v38 = a12;
      v37 = _mm_loadu_si128((const __m128i *)&a11);
    }
    a11 = (__int128)_mm_loadu_si128(&v37);
    a12 = v38;
  }
  if ( (_BYTE)a13 )
  {
    v23 = sub_1D13440(a13);
  }
  else
  {
    v33 = a1[4];
    v27 = sub_1F58D40(&a13, a2, a3, v22, a5, a6);
    v24 = v33;
    v23 = v27;
  }
  v25 = sub_1E0B8E0(v24, v35, (unsigned int)(v23 + 7) >> 3, a15, a17, a18, a11, a12, 1, 0, 0);
  return sub_1D260A0(a1, v20, v19, a4, a5, v18, a7, a8, a9, a10, a13, a14, v25);
}
