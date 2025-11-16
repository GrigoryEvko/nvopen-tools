// Function: sub_2AA7AA0
// Address: 0x2aa7aa0
//
bool __fastcall sub_2AA7AA0(__int64 a1, __int64 *a2)
{
  int v2; // edx

  sub_DFB6F0(
    *(__int64 **)(**(_QWORD **)a1 + 24LL),
    (unsigned int)***(unsigned __int8 ***)(*(_QWORD *)a1 + 8LL) - 29,
    *(_QWORD *)(**(_QWORD **)(*(_QWORD *)a1 + 16LL) + 8LL),
    *(_QWORD *)(**(_QWORD **)(*(_QWORD *)a1 + 24LL) + 8LL),
    *(_QWORD *)(**(_QWORD **)(*(_QWORD *)a1 + 32LL) + 8LL),
    *a2);
  return v2 == 0;
}
