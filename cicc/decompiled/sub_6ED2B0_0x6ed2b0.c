// Function: sub_6ED2B0
// Address: 0x6ed2b0
//
__int64 __fastcall sub_6ED2B0(__int64 a1)
{
  __int64 v1; // rax

  if ( *(_BYTE *)(a1 + 16) == 1 && (v1 = *(_QWORD *)(a1 + 144), *(_BYTE *)(v1 + 24) == 3) )
    return sub_6EA380(*(_QWORD *)(v1 + 56), 0, 0, 0);
  else
    return 0;
}
