// Function: sub_869970
// Address: 0x869970
//
__int64 __fastcall sub_869970(__int64 a1)
{
  __int64 result; // rax

  result = qword_4F04C68[0] + 776LL * dword_4F04C64;
  if ( *(_QWORD *)(result + 328) )
  {
    **(_QWORD **)(result + 336) = a1;
    result = qword_4F04C68[0] + 776LL * dword_4F04C64;
    *(_QWORD *)(a1 + 8) = *(_QWORD *)(result + 336);
  }
  else
  {
    *(_QWORD *)(result + 328) = a1;
  }
  *(_QWORD *)(result + 336) = a1;
  return result;
}
