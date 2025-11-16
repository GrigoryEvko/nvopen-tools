// Function: sub_1273520
// Address: 0x1273520
//
__int64 __fastcall sub_1273520(_QWORD *a1, const __m128i *a2, __int64 a3, int a4)
{
  __int64 v6; // rsi
  int v7; // ebx
  __int64 v8; // rax
  __int64 result; // rax

  v6 = a3;
  if ( *(char *)(a3 + 142) >= 0 && *(_BYTE *)(a3 + 140) == 12 )
  {
    v6 = a3;
    v7 = sub_8D4AB0(a3);
  }
  else
  {
    v7 = *(_DWORD *)(a3 + 136);
  }
  v8 = sub_127A040(a1 + 1, v6);
  result = sub_15A9FE0(a1[46], v8);
  if ( (_DWORD)result != v7 )
    return sub_1273260(a1, a2, (__int64)"align", (a4 << 16) | (unsigned int)(unsigned __int16)v7);
  return result;
}
