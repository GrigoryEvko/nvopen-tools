// Function: sub_CFE790
// Address: 0xcfe790
//
__int64 __fastcall sub_CFE790(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 *a6)
{
  void (*v6)(void); // rax

  v6 = *(void (**)(void))(*(_QWORD *)a1 + 128LL);
  if ( (char *)v6 == (char *)sub_CFE780 )
  {
    if ( (_BYTE)qword_4F866C8 )
      sub_CFE4C0(a1, a2, (__int64)sub_CFE780, a4, a5, a6);
    return 0;
  }
  else
  {
    v6();
    return 0;
  }
}
