// Function: sub_2EAC2B0
// Address: 0x2eac2b0
//
__int64 __fastcall sub_2EAC2B0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  int v3; // edx

  v2 = sub_2F3F520(*(_QWORD *)(a2 + 352));
  *(_BYTE *)(a1 + 20) = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)a1 = v2 | 4;
  v3 = 0;
  if ( v2 )
    v3 = *(_DWORD *)(v2 + 12);
  *(_DWORD *)(a1 + 16) = v3;
  return a1;
}
