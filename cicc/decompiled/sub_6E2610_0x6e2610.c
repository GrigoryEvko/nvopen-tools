// Function: sub_6E2610
// Address: 0x6e2610
//
__int64 __fastcall sub_6E2610(__int64 a1, char a2, int a3)
{
  __int64 result; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rdi
  _DWORD v8[9]; // [rsp+Ch] [rbp-24h] BYREF

  result = qword_4D03C50;
  if ( qword_4D03C50 && *(_QWORD *)(qword_4D03C50 + 120LL) )
  {
    if ( a3 )
    {
      sub_7296C0(v8);
      v5 = sub_727640();
      v6 = qword_4D03C50;
      *(_BYTE *)(v5 + 8) = a2;
      *(_QWORD *)(v5 + 16) = a1;
      **(_QWORD **)(v6 + 120) = v5;
      v7 = v8[0];
      *(_QWORD *)(qword_4D03C50 + 120LL) = v5;
      return sub_729730(v7);
    }
    else
    {
      result = sub_727640();
      v4 = qword_4D03C50;
      *(_BYTE *)(result + 8) = a2;
      *(_QWORD *)(result + 16) = a1;
      **(_QWORD **)(v4 + 120) = result;
      *(_QWORD *)(qword_4D03C50 + 120LL) = result;
    }
  }
  return result;
}
