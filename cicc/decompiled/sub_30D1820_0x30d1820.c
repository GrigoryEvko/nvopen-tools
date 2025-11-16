// Function: sub_30D1820
// Address: 0x30d1820
//
__int64 __fastcall sub_30D1820(_DWORD *a1, __int64 a2)
{
  _QWORD *v2; // rsi
  unsigned __int64 v4; // rdi
  int v5; // eax
  __int64 v6; // rdi
  __int64 result; // rax

  v2 = (_QWORD *)(a2 + 48);
  v4 = *v2 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (_QWORD *)v4 == v2 )
  {
    v6 = 0;
  }
  else
  {
    if ( !v4 )
      BUG();
    v5 = *(unsigned __int8 *)(v4 - 24);
    v6 = v4 - 24;
    if ( (unsigned int)(v5 - 30) >= 0xB )
      v6 = 0;
  }
  if ( (unsigned int)sub_B46E30(v6) > 1 )
    a1[183] = 1;
  result = (unsigned int)a1[189];
  a1[190] -= result;
  return result;
}
