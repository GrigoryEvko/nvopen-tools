// Function: sub_2DB2060
// Address: 0x2db2060
//
__int64 __fastcall sub_2DB2060(__int64 a1, __int64 a2)
{
  __int64 (*v2)(void); // rax

  v2 = *(__int64 (**)(void))(*(_QWORD *)a1 + 992LL);
  if ( (char *)v2 == (char *)sub_2DB1B50 )
    return (*(_QWORD *)(*(_QWORD *)(a2 + 16) + 24LL) >> 22) & 1LL;
  else
    return v2();
}
