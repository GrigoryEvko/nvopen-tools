// Function: sub_108E1C0
// Address: 0x108e1c0
//
__int64 __fastcall sub_108E1C0(__int64 a1, __int64 *a2, __int16 a3, char a4)
{
  sub_108DF50(
    a1,
    *(char **)(*a2 + 160),
    *(_QWORD *)(*a2 + 168),
    a2[2],
    a3,
    *(_WORD *)(*(_QWORD *)(*a2 + 152) + 48LL),
    a4,
    1);
  return sub_108CFF0(
           a1,
           a2[3],
           *(_BYTE *)(*a2 + 149) | (unsigned __int8)(8 * *(_BYTE *)(*a2 + 32)),
           *(_BYTE *)(*a2 + 148));
}
