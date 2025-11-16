// Function: sub_343FD80
// Address: 0x343fd80
//
__int64 __fastcall sub_343FD80(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rcx
  __int64 (*v5)(); // r8

  v4 = *(_QWORD *)(**(_QWORD **)(a4 + 40) + 120LL);
  v5 = *(__int64 (**)())(*(_QWORD *)a1 + 200LL);
  if ( v5 == sub_2FE2F30
    || !((unsigned __int8 (__fastcall *)(__int64, _QWORD, _QWORD, __int64))v5)(
          a1,
          **(unsigned __int16 **)(a2 + 48),
          *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL),
          v4) )
  {
    return 0;
  }
  else
  {
    return a2;
  }
}
