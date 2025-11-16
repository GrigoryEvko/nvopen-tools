// Function: sub_3981A60
// Address: 0x3981a60
//
__int64 __fastcall sub_3981A60(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v4; // r13
  __int64 v5; // r13
  __int64 v6; // rbx
  unsigned __int16 *v7; // rdi

  sub_16BD430(a2, *(unsigned __int16 *)(a1 + 12));
  result = sub_16BD430(a2, *(unsigned __int8 *)(a1 + 14));
  v4 = *(unsigned int *)(a1 + 24);
  if ( (_DWORD)v4 )
  {
    v5 = 16 * v4;
    v6 = 0;
    do
    {
      v7 = (unsigned __int16 *)(v6 + *(_QWORD *)(a1 + 16));
      v6 += 16;
      result = sub_3981A20(v7, a2);
    }
    while ( v6 != v5 );
  }
  return result;
}
