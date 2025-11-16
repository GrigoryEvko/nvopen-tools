// Function: sub_16433E0
// Address: 0x16433e0
//
void *__fastcall sub_16433E0(__int64 a1, __int64 *a2, const void *a3, unsigned int a4, unsigned __int8 a5)
{
  __int64 v6; // rax
  __int64 v7; // rax
  void *result; // rax

  v6 = *a2;
  *(_QWORD *)(a1 + 24) = a2;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)a1 = v6;
  v7 = *(_QWORD *)(a1 + 8) & 0xFFFFFF00LL | 0xC;
  *(_QWORD *)(a1 + 8) = v7;
  result = (void *)((a5 << 8) | (unsigned int)(v7 & 0xC));
  *(_DWORD *)(a1 + 8) = (_DWORD)result;
  if ( a4 )
    result = memcpy((void *)(a1 + 32), a3, 8LL * a4);
  *(_QWORD *)(a1 + 16) = a1 + 24;
  *(_DWORD *)(a1 + 12) = a4 + 1;
  return result;
}
