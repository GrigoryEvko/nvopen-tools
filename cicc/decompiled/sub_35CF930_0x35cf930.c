// Function: sub_35CF930
// Address: 0x35cf930
//
__int64 __fastcall sub_35CF930(_QWORD *a1, __int64 *a2)
{
  __int64 v4; // rsi
  __int64 v5; // rcx
  __int64 v6; // rdi
  __int64 v7; // r8
  __int64 (*v8)(); // rax
  __int64 (*v9)(); // rax
  __int64 v11; // rax
  int v12; // eax

  v4 = *a2;
  if ( (unsigned __int8)sub_BB98D0(a1, v4) || (__int64 *)(a2[40] & 0xFFFFFFFFFFFFFFF8LL) == a2 + 40 )
    return 0;
  v6 = a2[2];
  v7 = 0;
  v8 = *(__int64 (**)())(*(_QWORD *)v6 + 136LL);
  if ( v8 != sub_2DD19D0 )
    v7 = ((__int64 (__fastcall *)(__int64, __int64, __int64 *, __int64, _QWORD))v8)(v6, v4, a2 + 40, v5, 0);
  if ( (_DWORD)qword_50400C8 == 1 )
    return sub_35CC920(a1, (__int64)a2);
  if ( (_DWORD)qword_50400C8 == 2 )
    return 0;
  if ( (_DWORD)qword_50400C8 )
    BUG();
  v9 = *(__int64 (**)())(*(_QWORD *)v7 + 72LL);
  if ( v9 == sub_2FDBBA0 )
    return 0;
  if ( !((unsigned __int8 (__fastcall *)(__int64, __int64 *))v9)(v7, a2) )
    return 0;
  v11 = *(_QWORD *)(a2[1] + 656);
  if ( *(_DWORD *)(v11 + 336) == 4 )
  {
    v12 = *(_DWORD *)(v11 + 344);
    if ( v12 )
    {
      if ( v12 != 6 )
        return 0;
    }
  }
  if ( (unsigned __int8)sub_B2D610(*a2, 56)
    || (unsigned __int8)sub_B2D610(*a2, 63)
    || (unsigned __int8)sub_B2D610(*a2, 59)
    || (unsigned __int8)sub_B2D610(*a2, 64)
    || (unsigned __int8)sub_B2D610(*a2, 57) )
  {
    return 0;
  }
  else
  {
    return sub_35CC920(a1, (__int64)a2);
  }
}
