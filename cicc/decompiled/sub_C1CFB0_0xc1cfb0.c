// Function: sub_C1CFB0
// Address: 0xc1cfb0
//
__int64 __fastcall sub_C1CFB0(unsigned __int64 *a1, unsigned __int64 *a2, unsigned __int64 a3)
{
  __int64 v4; // rax
  bool v5; // cf
  unsigned __int64 v6; // r15
  unsigned __int64 v7; // r12
  unsigned __int64 *v8; // rbx
  int v9; // ecx
  unsigned int i; // [rsp+Ch] [rbp-64h]
  bool v12; // [rsp+2Fh] [rbp-41h] BYREF
  __m128i v13[4]; // [rsp+30h] [rbp-40h] BYREF

  v4 = sub_C1B1E0(*a2, a3, *a1, (bool *)v13[0].m128i_i8);
  v5 = v13[0].m128i_i8[0] == 0;
  *a1 = v4;
  v6 = a2[3];
  for ( i = v5 ? 0 : 0xA; v6; v6 = *(_QWORD *)v6 )
  {
    v7 = *(_QWORD *)(v6 + 24);
    v13[0] = _mm_loadu_si128((const __m128i *)(v6 + 8));
    v8 = sub_C1CD30(a1 + 1, v13);
    *v8 = sub_C1B1E0(v7, a3, *v8, &v12);
    if ( v12 )
    {
      v9 = 10;
      if ( i )
        v9 = i;
      i = v9;
    }
  }
  return i;
}
