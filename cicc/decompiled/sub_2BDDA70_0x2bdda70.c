// Function: sub_2BDDA70
// Address: 0x2bdda70
//
bool __fastcall sub_2BDDA70(_QWORD **a1, char *a2)
{
  unsigned int v2; // r13d
  __int64 v3; // rax
  __int64 v5; // rax

  v2 = *a2;
  if ( !byte_5010628[0] && (unsigned int)sub_2207590((__int64)byte_5010628) )
  {
    v5 = sub_222F790(*a1, (__int64)a2);
    a2 = 0;
    unk_5010630 = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v5 + 32LL))(v5, 0);
    sub_2207640((__int64)byte_5010628);
  }
  v3 = sub_222F790(*a1, (__int64)a2);
  return unk_5010630 != (*(unsigned __int8 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v3 + 32LL))(v3, v2);
}
