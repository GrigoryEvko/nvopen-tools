// Function: sub_1CA28F0
// Address: 0x1ca28f0
//
_BYTE *__fastcall sub_1CA28F0(_QWORD *a1, __int64 a2, _BYTE *a3, __int64 a4, _QWORD *a5, int a6, unsigned __int8 a7)
{
  __int64 v7; // rax
  int v8; // r9d

  v7 = (unsigned int)(a6 - 1);
  v8 = 0;
  if ( (unsigned int)v7 <= 0xF )
    v8 = dword_42DFC80[v7];
  return sub_1CA1B70(a1, a2, a3, a4, a5, v8, a7);
}
