// Function: sub_2FE0330
// Address: 0x2fe0330
//
bool __fastcall sub_2FE0330(__int64 a1, int a2, int a3)
{
  bool result; // al
  __int64 (*v4)(void); // rax
  __int64 v6; // rax

  result = 1;
  if ( a2 != a3 )
  {
    v4 = *(__int64 (**)(void))(*(_QWORD *)a1 + 648LL);
    if ( v4 == sub_2FDC5D0 )
    {
      return 0;
    }
    else
    {
      v6 = v4();
      return BYTE4(v6) && (_DWORD)v6 == a3;
    }
  }
  return result;
}
