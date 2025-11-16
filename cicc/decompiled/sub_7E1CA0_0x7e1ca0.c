// Function: sub_7E1CA0
// Address: 0x7e1ca0
//
__int64 __fastcall sub_7E1CA0(__int64 a1)
{
  __int64 result; // rax

  result = unk_4F07288;
  *(_QWORD *)(a1 + 112) = *(_QWORD *)(unk_4F07288 + 104LL);
  *(_QWORD *)(result + 104) = a1;
  if ( !dword_4F07588 || *(_QWORD *)(a1 + 112) )
  {
    if ( *(_QWORD *)(a1 + 40) )
      return result;
LABEL_5:
    *(_QWORD *)(a1 + 40) = result;
    return result;
  }
  *(_QWORD *)(qword_4D03FF0 + 56) = a1;
  if ( !*(_QWORD *)(a1 + 40) )
    goto LABEL_5;
  return result;
}
