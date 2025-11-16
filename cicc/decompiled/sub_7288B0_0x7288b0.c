// Function: sub_7288B0
// Address: 0x7288b0
//
void __fastcall sub_7288B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rdi
  __int64 v8; // rdi

  v7 = *(_QWORD *)(a1 + 16);
  if ( v7 && !(unsigned int)sub_8D7760(v7, a2, a3, a4, a5, a6)
    || *(_BYTE *)(a1 + 48) == 5
    && ((v8 = *(_QWORD *)(a1 + 56)) == 0 || !(unsigned int)sub_8D7760(v8, a2, a3, a4, a5, a6)) )
  {
    *(_DWORD *)(a2 + 80) = 1;
    *(_DWORD *)(a2 + 72) = 1;
  }
}
