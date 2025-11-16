// Function: sub_144A7C0
// Address: 0x144a7c0
//
__int64 __fastcall sub_144A7C0(__int64 a1, _QWORD *a2)
{
  __int64 v2; // rax
  __int64 result; // rax

  v2 = a2[1];
  if ( *a2 != v2 )
  {
    while ( (*(int (__fastcall **)(_QWORD))(**(_QWORD **)(v2 - 8) + 40LL))(*(_QWORD *)(v2 - 8)) > 5 )
    {
      sub_160FB80(a2);
      v2 = a2[1];
      if ( *a2 == v2 )
        goto LABEL_6;
    }
    v2 = a2[1];
  }
LABEL_6:
  result = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(v2 - 8) + 40LL))(*(_QWORD *)(v2 - 8));
  if ( (_DWORD)result == 5 )
  {
    result = sub_1614540(*(_QWORD *)(a2[1] - 8LL), a1);
    if ( !(_BYTE)result )
      return sub_160FB80(a2);
  }
  return result;
}
