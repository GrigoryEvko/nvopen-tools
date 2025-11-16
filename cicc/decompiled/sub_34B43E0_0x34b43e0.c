// Function: sub_34B43E0
// Address: 0x34b43e0
//
__int64 __fastcall sub_34B43E0(_QWORD *a1, unsigned int *a2, const __m128i *a3)
{
  __int64 v5; // rax
  unsigned int v6; // esi
  _QWORD *v7; // rdx
  _QWORD *v8; // r8
  __m128i v9; // xmm0
  __int64 v10; // r12
  unsigned int v11; // ecx
  _QWORD *v12; // rax
  char v14; // di

  v5 = sub_22077B0(0x38u);
  v6 = *a2;
  v7 = (_QWORD *)a1[2];
  v8 = a1 + 1;
  v9 = _mm_loadu_si128(a3);
  v10 = v5;
  *(_DWORD *)(v5 + 32) = v6;
  *(__m128i *)(v5 + 40) = v9;
  if ( v7 )
  {
    while ( 1 )
    {
      v11 = *((_DWORD *)v7 + 8);
      v12 = (_QWORD *)v7[3];
      if ( v6 < v11 )
        v12 = (_QWORD *)v7[2];
      if ( !v12 )
        break;
      v7 = v12;
    }
    v14 = 1;
    if ( v8 != v7 )
      v14 = v6 < v11;
  }
  else
  {
    v7 = a1 + 1;
    v14 = 1;
  }
  sub_220F040(v14, v10, v7, a1 + 1);
  ++a1[5];
  return v10;
}
