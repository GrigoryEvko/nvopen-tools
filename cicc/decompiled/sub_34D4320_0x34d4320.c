// Function: sub_34D4320
// Address: 0x34d4320
//
unsigned __int64 __fastcall sub_34D4320(__int64 a1, unsigned int a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r12
  __int64 v5; // rbx
  unsigned int v6; // eax
  bool v7; // of
  unsigned __int64 result; // rax

  v4 = a4;
  v5 = sub_34D3270(a1 + 8, a2, a3, *(_QWORD *)(a4 + 24), 0, 0, 0);
  if ( (unsigned int)*(unsigned __int8 *)(v4 + 8) - 17 <= 1 )
    v4 = **(_QWORD **)(v4 + 16);
  v6 = sub_34D06B0(a1 + 8, (__int64 *)v4);
  v7 = __OFADD__(v5, v6);
  result = v5 + v6;
  if ( v7 )
  {
    result = 0x7FFFFFFFFFFFFFFFLL;
    if ( v5 <= 0 )
      return 0x8000000000000000LL;
  }
  return result;
}
