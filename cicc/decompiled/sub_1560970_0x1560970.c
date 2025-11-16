// Function: sub_1560970
// Address: 0x1560970
//
__int64 __fastcall sub_1560970(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 result; // rax

  v2 = *(_QWORD *)(a2 + 88);
  result = a1;
  *(_DWORD *)a1 = HIDWORD(v2);
  if ( (_DWORD)v2 == -1 )
  {
    *(_BYTE *)(a1 + 8) = 0;
  }
  else
  {
    *(_BYTE *)(a1 + 8) = 1;
    *(_DWORD *)(a1 + 4) = v2;
  }
  return result;
}
