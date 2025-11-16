// Function: sub_32889F0
// Address: 0x32889f0
//
__int64 __fastcall sub_32889F0(
        __int64 a1,
        int a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        __int128 a8,
        __int128 a9)
{
  __m128i v11; // xmm0
  char v12; // r15
  __int64 v13; // rdx
  int v14; // r9d
  __int128 v16; // rax
  __int128 v17; // [rsp-10h] [rbp-C0h]
  __m128i v18; // [rsp+0h] [rbp-B0h] BYREF
  __int128 v19; // [rsp+10h] [rbp-A0h]
  _QWORD v20[2]; // [rsp+20h] [rbp-90h] BYREF
  __int16 v21; // [rsp+30h] [rbp-80h]
  __int64 v22; // [rsp+38h] [rbp-78h]
  _QWORD v23[4]; // [rsp+40h] [rbp-70h] BYREF
  __m128i v24; // [rsp+60h] [rbp-50h]
  __int64 v25; // [rsp+70h] [rbp-40h]
  __int64 v26; // [rsp+78h] [rbp-38h]

  v11 = _mm_loadu_si128((const __m128i *)&a7);
  v12 = BYTE8(a9);
  if ( *((_QWORD *)&a8 + 1) )
  {
    v23[2] = a5;
    v23[3] = a6;
    v24 = v11;
    v23[0] = *((_QWORD *)&a8 + 1);
    v23[1] = a9;
    v25 = sub_33ED040(a1, (unsigned int)a8);
    v26 = v13;
    *((_QWORD *)&v17 + 1) = 4;
    *(_QWORD *)&v17 = v23;
    v21 = 1;
    v20[0] = a3;
    v20[1] = a4;
    v22 = 0;
    return sub_3411BE0(a1, 147 - ((unsigned int)(v12 == 0) - 1), a2, (unsigned int)v20, 2, v14, v17);
  }
  else
  {
    *(_QWORD *)&v19 = a5;
    *((_QWORD *)&v19 + 1) = a6;
    v18 = v11;
    *(_QWORD *)&v16 = sub_33ED040(a1, (unsigned int)a8);
    return sub_340F900(a1, 208, a2, a3, a4, DWORD2(v19), v19, *(_OWORD *)&_mm_load_si128(&v18), v16);
  }
}
