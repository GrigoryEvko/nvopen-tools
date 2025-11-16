// Function: sub_39BFAD0
// Address: 0x39bfad0
//
__int64 __fastcall sub_39BFAD0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int16 v4; // ax
  __int64 v6; // rax

  if ( !byte_5057610 && (unsigned int)sub_2207590((__int64)&byte_5057610) )
  {
    v6 = sub_396DDB0(a2);
    byte_5057618 = sub_15A9520(v6, 0);
    sub_2207640((__int64)&byte_5057610);
  }
  (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(a2 + 256) + 160LL))(*(_QWORD *)(a2 + 256), a3, 0);
  sub_396F340(a2, *(_DWORD *)(a1 + 16) * (unsigned __int8)byte_5057618 + 4);
  v4 = sub_3971A70(a2);
  sub_396F320(a2, v4);
  sub_396F300(a2, (unsigned __int8)byte_5057618);
  return sub_396F300(a2, 0);
}
