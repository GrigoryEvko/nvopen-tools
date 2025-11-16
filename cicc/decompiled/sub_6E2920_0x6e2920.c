// Function: sub_6E2920
// Address: 0x6e2920
//
__int64 __fastcall sub_6E2920(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  __int64 v7; // r14
  __int64 v8; // rdx
  _QWORD v9[13]; // [rsp+0h] [rbp-100h] BYREF
  int v10; // [rsp+6Ch] [rbp-94h]

  result = qword_4D03C50;
  v7 = *(_QWORD *)(qword_4D03C50 + 48LL);
  if ( a1 )
  {
    if ( dword_4F077C4 == 2 )
    {
      v8 = (unsigned int)dword_4D03A90;
      if ( dword_4D03A90
        || (a4 = *(_QWORD *)(qword_4D03C50 + 64LL), v8 = *(_QWORD *)(qword_4F06BC0 + 24LL), v8 != a4)
        && a4 != *(_QWORD *)(v8 + 32) )
      {
        sub_76C7C0(v9, a2, v8, a4, a5, a6);
        a2 = v9;
        v9[0] = sub_6E0040;
        v9[4] = sub_6E0190;
        v9[5] = sub_6DF600;
        v10 = 1;
        sub_76D400(a1, v9);
        result = qword_4D03C50;
      }
      if ( (*(_BYTE *)(result + 20) & 4) != 0 )
      {
        sub_76C7C0(v9, a2, v8, a4, a5, a6);
        v9[0] = sub_6E5AE0;
        result = sub_76D400(a1, v9);
      }
      if ( dword_4F077C4 == 2 )
      {
        if ( v7 )
          return sub_732E60(v7, 30, a1);
      }
    }
  }
  else if ( v7 && dword_4F077C4 == 2 )
  {
    return sub_733CF0(*(_QWORD *)(qword_4D03C50 + 48LL), a2);
  }
  return result;
}
