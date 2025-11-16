// Function: sub_6E3AC0
// Address: 0x6e3ac0
//
__int64 __fastcall sub_6E3AC0(__int64 a1, __int64 *a2, int a3, _QWORD *a4)
{
  __int64 result; // rax
  __int64 v5; // r12
  __int64 v6; // rax
  _QWORD *v7; // [rsp-38h] [rbp-38h]
  int v8; // [rsp-2Ch] [rbp-2Ch]

  result = qword_4D03C50;
  if ( (*(_BYTE *)(qword_4D03C50 + 19LL) & 2) != 0 )
  {
    v5 = *(_QWORD *)(a1 + 80);
    if ( !v5 )
    {
      v7 = a4;
      v8 = a3;
      v5 = sub_823970(384);
      sub_6E2E50(0, v5);
      a4 = v7;
      *(_BYTE *)(v5 + 352) = 4;
      *(_DWORD *)(v5 + 364) = 0;
      a3 = v8;
      *(_QWORD *)(v5 + 376) = 0;
      v6 = *(_QWORD *)&dword_4F077C8;
      *(_QWORD *)(v5 + 356) = *(_QWORD *)&dword_4F077C8;
      *(_QWORD *)(v5 + 368) = v6;
      *(_QWORD *)(a1 + 80) = v5;
    }
    result = *a2;
    *(_DWORD *)(v5 + 364) = a3;
    *(_QWORD *)(v5 + 356) = result;
    if ( a4 )
    {
      result = *a4;
      *(_QWORD *)(v5 + 368) = *a4;
    }
  }
  return result;
}
