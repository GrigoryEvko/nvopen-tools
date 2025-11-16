// Function: sub_27DD030
// Address: 0x27dd030
//
bool __fastcall sub_27DD030(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // rax
  __int64 v3; // r12

  v2 = *(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v2 == a2 + 48 )
    return 0;
  if ( !v2 )
    BUG();
  v3 = v2 - 24;
  return (unsigned int)*(unsigned __int8 *)(v2 - 24) - 30 <= 0xA && (unsigned int)sub_B46E30(v3) > 1 && sub_BC8A50(v3);
}
