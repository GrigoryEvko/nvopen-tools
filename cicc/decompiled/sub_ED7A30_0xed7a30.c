// Function: sub_ED7A30
// Address: 0xed7a30
//
__int64 __fastcall sub_ED7A30(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rdi

  *(_QWORD *)a1 = &unk_49E4DB8;
  sub_C7D6A0(*(_QWORD *)(a1 + 32), 24LL * *(unsigned int *)(a1 + 48), 8);
  result = sub_EE5E50(a1 + 16);
  v2 = *(_QWORD *)(a1 + 8);
  if ( v2 )
    return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v2 + 8LL))(v2);
  return result;
}
