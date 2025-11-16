// Function: sub_2FE04D0
// Address: 0x2fe04d0
//
__int64 __fastcall sub_2FE04D0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 (*v3)(); // rax
  __int64 (*v6)(); // rax

  v3 = *(__int64 (**)())(*(_QWORD *)a1 + 640LL);
  if ( v3 == sub_2FDC5C0 )
    return 0;
  if ( ((unsigned __int8 (__fastcall *)(__int64, __int64, _QWORD))v3)(a1, a2, 0)
    || (v6 = *(__int64 (**)())(*(_QWORD *)a1 + 640LL), v6 != sub_2FDC5C0)
    && ((unsigned __int8 (__fastcall *)(__int64, __int64, __int64))v6)(a1, a2, 1) )
  {
    if ( (*(unsigned __int8 (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)a1 + 656LL))(
           a1,
           a2,
           *(_QWORD *)(a2 + 24)) )
    {
      return (*(__int64 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)a1 + 664LL))(a1, a2, a3);
    }
  }
  return 0;
}
