// Function: sub_7B0C00
// Address: 0x7b0c00
//
__int64 sub_7B0C00()
{
  __int64 v0; // rdi
  int v1; // eax
  __m128i *v2; // rdx
  __int64 result; // rax
  int v4; // ebx
  __int64 v5; // r12
  __int64 v6; // rax

  v0 = qword_4F17FE0;
  v1 = dword_4F17FD8 + 1;
  if ( dword_4F17FD8 + 1 == dword_4F17FDC )
  {
    v4 = dword_4F17FD8 + 31;
    v5 = qword_4F17FD0 - qword_4F17FE0;
    v6 = sub_822C60(qword_4F17FE0, 112LL * v4 - 3360, 112LL * v4);
    dword_4F17FDC = v4;
    qword_4F17FE0 = v6;
    v0 = v6;
    qword_4F17FD0 = v6 + v5;
    v1 = dword_4F17FD8 + 1;
  }
  dword_4F17FD8 = v1;
  v2 = (__m128i *)(v0 + 112LL * v1);
  unk_4F064B0 = v2;
  *v2 = _mm_loadu_si128(v2 - 7);
  v2[1] = _mm_loadu_si128(v2 - 6);
  v2[2] = _mm_loadu_si128(v2 - 5);
  v2[3] = _mm_loadu_si128(v2 - 4);
  v2[4] = _mm_loadu_si128(v2 - 3);
  v2[5] = _mm_loadu_si128(v2 - 2);
  result = unk_4F064B0;
  v2[6] = _mm_loadu_si128(v2 - 1);
  *(_BYTE *)(result + 88) |= 0x40u;
  return result;
}
