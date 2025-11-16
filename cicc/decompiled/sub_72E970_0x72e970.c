// Function: sub_72E970
// Address: 0x72e970
//
__int64 __fastcall sub_72E970(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax

  result = sub_72DB90(a1, a2, a3, a4, a5, a6);
  *(_DWORD *)(a2 + 76) = 1;
  dword_4F07AD0 += result;
  return result;
}
