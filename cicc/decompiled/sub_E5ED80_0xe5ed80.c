// Function: sub_E5ED80
// Address: 0xe5ed80
//
bool __fastcall sub_E5ED80(_QWORD *a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v3; // rax

  v2 = *(_QWORD *)(a2 + 48);
  v3 = sub_E66210(*a1);
  sub_E60660(v3, a1, a2);
  return *(_QWORD *)(a2 + 48) != (unsigned int)v2;
}
