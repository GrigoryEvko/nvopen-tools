// Function: sub_2F28E80
// Address: 0x2f28e80
//
_BOOL8 __fastcall sub_2F28E80(__int64 a1, unsigned int a2)
{
  _BOOL8 result; // rax
  _QWORD *v3; // r13

  result = 0;
  if ( a2 - 1 <= 0x3FFFFFFE )
  {
    v3 = *(_QWORD **)(a1 + 24);
    if ( !*(_BYTE *)(*(_QWORD *)(*(_QWORD *)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(*v3 + 16LL) + 200LL))(*(_QWORD *)(*v3 + 16LL))
                                           + 248)
                               + 16LL)
                   + a2)
      || (*(_QWORD *)(v3[48] + 8LL * (a2 >> 6)) & (1LL << a2)) != 0 )
    {
      return 1;
    }
  }
  return result;
}
