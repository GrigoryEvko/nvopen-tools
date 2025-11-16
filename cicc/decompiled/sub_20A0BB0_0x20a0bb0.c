// Function: sub_20A0BB0
// Address: 0x20a0bb0
//
__int64 __fastcall sub_20A0BB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rdi
  __int64 (*v5)(); // r8

  v4 = *(_QWORD *)(a4 + 16);
  v5 = *(__int64 (**)())(*(_QWORD *)v4 + 80LL);
  if ( v5 == sub_1F3C990
    || !((unsigned __int8 (__fastcall *)(__int64, _QWORD, _QWORD, _QWORD))v5)(
          v4,
          **(unsigned __int8 **)(a2 + 40),
          *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL),
          *(_QWORD *)(**(_QWORD **)(a4 + 32) + 112LL)) )
  {
    return 0;
  }
  else
  {
    return a2;
  }
}
