// Function: sub_B14810
// Address: 0xb14810
//
__int64 __fastcall sub_B14810(__int64 a1, __int64 a2)
{
  __int64 v2; // rax

  if ( *(_QWORD *)(a1 + 16) )
  {
    v2 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a2 + 48LL))(a2);
    (*(void (__fastcall **)(__int64, char *))(*(_QWORD *)v2 + 48LL))(v2, ": ");
  }
  return (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a2 + 128LL))(a2, *(_QWORD *)(a1 + 24));
}
