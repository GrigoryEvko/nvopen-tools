// Function: sub_15A95A0
// Address: 0x15a95a0
//
__int64 __fastcall sub_15A95A0(__int64 a1, unsigned int a2)
{
  __int64 v2; // rax

  v2 = sub_15A8580(a1, a2);
  if ( v2 == *(_QWORD *)(a1 + 224) + 20LL * *(unsigned int *)(a1 + 232) || *(_DWORD *)(v2 + 12) != a2 )
    v2 = sub_15A8580(a1, 0);
  return *(unsigned int *)(v2 + 16);
}
