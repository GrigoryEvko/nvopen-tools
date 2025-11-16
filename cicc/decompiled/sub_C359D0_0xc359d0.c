// Function: sub_C359D0
// Address: 0xc359d0
//
__int64 __fastcall sub_C359D0(__int64 a1, char a2)
{
  __int64 v2; // rdx
  unsigned int v3; // r13d
  __int64 v4; // rax

  v2 = *(_QWORD *)a1;
  if ( a2 && !*(_BYTE *)(v2 + 25) )
    BUG();
  *(_BYTE *)(a1 + 20) = *(_BYTE *)(a1 + 20) & 0xF0 | 2 | (8 * (a2 & 1));
  *(_DWORD *)(a1 + 16) = *(_DWORD *)(v2 + 4);
  v3 = sub_C337D0(a1);
  v4 = sub_C33900(a1);
  return sub_C45D00(v4, 1, v3);
}
