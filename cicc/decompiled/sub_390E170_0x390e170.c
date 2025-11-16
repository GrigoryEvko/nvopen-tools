// Function: sub_390E170
// Address: 0x390e170
//
__int64 __fastcall sub_390E170(__int64 a1, __int64 a2, __int64 a3, _QWORD *a4)
{
  return *(_QWORD *)(a1 + 16)
       * ((unsigned __int64)(*(_QWORD *)(a1 + 16) + a3 + sub_390E130(a1, a2, a4))
        / *(_QWORD *)(a1 + 16))
       - a3;
}
