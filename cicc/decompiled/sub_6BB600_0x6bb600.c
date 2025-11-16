// Function: sub_6BB600
// Address: 0x6bb600
//
_QWORD *__fastcall sub_6BB600(__int64 a1, unsigned int a2)
{
  _QWORD *result; // rax
  __int64 v3; // rbx
  __int64 v4; // rax
  bool v5; // zf
  _QWORD v6[5]; // [rsp+8h] [rbp-28h] BYREF

  if ( !qword_4D03C50 || (result = *(_QWORD **)(qword_4D03C50 + 136LL)) == 0 || !*result )
  {
    for ( result = (_QWORD *)sub_869470(v6); (_DWORD)result; result = (_QWORD *)sub_866C00(v6[0]) )
    {
      v3 = sub_6BB5A0(a2, 0);
      sub_6E1C20(v3, 0, a1);
      v4 = sub_867630(v6[0], 0);
      if ( v4 )
      {
        v5 = *(_BYTE *)(v3 + 8) == 0;
        *(_QWORD *)(v3 + 16) = v4;
        if ( v5 )
          *(_QWORD *)(*(_QWORD *)(v3 + 24) + 136LL) = v4;
      }
    }
  }
  return result;
}
