// Function: sub_2617880
// Address: 0x2617880
//
__int64 __fastcall sub_2617880(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rdi

  v2 = sub_B82360(*(_QWORD *)(*(_QWORD *)a1 + 8LL), (__int64)&unk_4F8662C);
  if ( v2 && (v3 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v2 + 104LL))(v2, &unk_4F8662C)) != 0 )
    return sub_CFB8F0(v3, a2);
  else
    return 0;
}
