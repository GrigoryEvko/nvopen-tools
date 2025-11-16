// Function: sub_1DD29D0
// Address: 0x1dd29d0
//
__int64 __fastcall sub_1DD29D0(__int64 *a1)
{
  __int64 v1; // r8
  __int64 v2; // rdx
  int v3; // ecx
  unsigned int v4; // esi
  __m128i v5; // xmm0
  __int64 result; // rax

  v1 = *a1;
  v2 = a1[1];
  v3 = *((_DWORD *)a1 + 4);
  v4 = *((_DWORD *)a1 + 5);
  while ( *(a1 - 2) > v2
       || *(a1 - 2) == v2 && (*((_DWORD *)a1 - 2) > v3 || *((_DWORD *)a1 - 2) == v3 && v4 < *((_DWORD *)a1 - 1)) )
  {
    v5 = _mm_loadu_si128((const __m128i *)(a1 - 3));
    result = *(a1 - 1);
    a1 -= 3;
    *(__m128i *)(a1 + 3) = v5;
    a1[5] = result;
  }
  *a1 = v1;
  a1[1] = v2;
  *((_DWORD *)a1 + 4) = v3;
  *((_DWORD *)a1 + 5) = v4;
  return result;
}
