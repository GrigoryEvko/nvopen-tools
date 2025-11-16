// Function: sub_CFC250
// Address: 0xcfc250
//
__int64 __fastcall sub_CFC250(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi

  v3 = a1 + 176;
  *(_QWORD *)(v3 - 176) = &unk_49DDB38;
  sub_CFBE20(v3, a2);
  sub_C7D6A0(*(_QWORD *)(a1 + 184), 48LL * *(unsigned int *)(a1 + 200), 8);
  return sub_BB9280(a1);
}
