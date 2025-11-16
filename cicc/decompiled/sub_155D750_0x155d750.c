// Function: sub_155D750
// Address: 0x155d750
//
__int64 __fastcall sub_155D750(__int64 a1, __int64 *a2)
{
  __int64 v2; // rax

  v2 = sub_155D4A0(*a2);
  *(_DWORD *)a1 = HIDWORD(v2);
  if ( (_DWORD)v2 == -1 )
  {
    *(_BYTE *)(a1 + 8) = 0;
    return a1;
  }
  else
  {
    *(_DWORD *)(a1 + 4) = v2;
    *(_BYTE *)(a1 + 8) = 1;
    return a1;
  }
}
