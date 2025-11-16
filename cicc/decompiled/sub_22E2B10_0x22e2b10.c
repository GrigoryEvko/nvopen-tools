// Function: sub_22E2B10
// Address: 0x22e2b10
//
__int64 __fastcall sub_22E2B10(__int64 *a1, _QWORD *a2)
{
  __int64 v2; // rax
  __int64 result; // rax

  v2 = a2[1];
  if ( *a2 != v2 )
  {
    while ( (*(int (__fastcall **)(_QWORD))(**(_QWORD **)(v2 - 8) + 40LL))(*(_QWORD *)(v2 - 8)) > 5 )
    {
      sub_B823C0((__int64)a2);
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
    result = sub_B88720(*(_QWORD *)(a2[1] - 8LL), a1);
    if ( !(_BYTE)result )
      return sub_B823C0((__int64)a2);
  }
  return result;
}
