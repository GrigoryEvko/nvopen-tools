// Function: sub_B14AD0
// Address: 0xb14ad0
//
__int64 __fastcall sub_B14AD0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // rax

  result = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a2 + 128LL))(a2, *(_QWORD *)(a1 + 24));
  if ( *(_QWORD *)(a1 + 16) )
  {
    v3 = (*(__int64 (__fastcall **)(__int64, char *))(*(_QWORD *)a2 + 48LL))(a2, " at line ");
    return (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v3 + 64LL))(v3, *(_QWORD *)(a1 + 16));
  }
  return result;
}
