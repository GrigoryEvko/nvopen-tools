// Function: sub_3258530
// Address: 0x3258530
//
__int64 __fastcall sub_3258530(__int64 a1)
{
  __int64 result; // rax

  if ( *(_BYTE *)(a1 + 28) && *(_QWORD *)(a1 + 32) && (*(_BYTE *)(a1 + 26) || *(_BYTE *)(a1 + 24)) )
    return (*(__int64 (__fastcall **)(_QWORD, _QWORD))(**(_QWORD **)(*(_QWORD *)(a1 + 8) + 224LL) + 1072LL))(
             *(_QWORD *)(*(_QWORD *)(a1 + 8) + 224LL),
             0);
  return result;
}
