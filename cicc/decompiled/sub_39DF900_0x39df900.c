// Function: sub_39DF900
// Address: 0x39df900
//
char __fastcall sub_39DF900(__int64 a1, __int64 a2)
{
  __int64 (*v2)(); // rax
  __int64 v3; // rax
  char result; // al

  v2 = *(__int64 (**)())(**(_QWORD **)(a2 + 16) + 48LL);
  if ( v2 == sub_1D90020 )
    JUMPOUT(0x4392C4);
  v3 = v2();
  result = (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v3 + 136LL))(v3, a2);
  if ( !result )
  {
    result = sub_15602E0((_QWORD *)(*(_QWORD *)a2 + 112LL), "no-frame-pointer-elim-non-leaf", 0x1Eu);
    if ( result )
      return *(_BYTE *)(*(_QWORD *)(a2 + 56) + 65LL);
  }
  return result;
}
