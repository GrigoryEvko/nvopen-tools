// Function: sub_396BE80
// Address: 0x396be80
//
__int64 __fastcall sub_396BE80(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  result = *(_QWORD *)(a1 + 240);
  if ( *(_BYTE *)(result + 307) )
    return sub_396BDC0(a1, a2);
  return result;
}
