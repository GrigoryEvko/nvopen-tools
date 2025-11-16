// Function: sub_88EF30
// Address: 0x88ef30
//
__int64 __fastcall sub_88EF30(__int64 a1)
{
  _QWORD *v1; // r12
  __int64 result; // rax

  v1 = (_QWORD *)sub_7C68A0(
                   *(__int64 **)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 88LL) + 104LL),
                   (FILE *)&dword_4F063F8,
                   1,
                   (*(_BYTE *)(a1 + 57) & 2) != 0);
  result = sub_8794A0(v1);
  *(_QWORD *)(a1 + 80) = v1;
  if ( (*(_BYTE *)(result + 160) & 2) != 0 || (*(_BYTE *)(result + 266) & 1) != 0 )
    *(_WORD *)(a1 + 56) |= 0x202u;
  return result;
}
