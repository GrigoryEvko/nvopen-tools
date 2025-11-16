// Function: sub_76FFC0
// Address: 0x76ffc0
//
__int64 __fastcall sub_76FFC0(__int64 a1)
{
  __int64 v1; // rsi
  __int64 v2; // rdx
  _QWORD *v3; // rcx
  _QWORD *i; // rax
  __int64 result; // rax

  v1 = *(_QWORD *)(a1 + 16);
  v2 = 2;
  v3 = *(_QWORD **)v1;
  for ( i = **(_QWORD ***)v1; i; ++v2 )
  {
    v3 = i;
    i = (_QWORD *)*i;
  }
  *v3 = qword_4F08088;
  *(_BYTE *)(a1 + 8) &= ~4u;
  result = *(_QWORD *)(v1 + 24);
  qword_4F08080 += v2;
  qword_4F08088 = v1;
  *(_QWORD *)(a1 + 16) = result;
  return result;
}
