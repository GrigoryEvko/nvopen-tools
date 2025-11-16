// Function: sub_71B580
// Address: 0x71b580
//
__int64 __fastcall sub_71B580(__int64 a1, __int64 a2, unsigned int *a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r12
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 result; // rax
  __int64 v11; // r13

  v5 = a1;
  *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(a2 + 80) + 80LL) + 24LL) &= ~1u;
  if ( (*(_BYTE *)(a1 + 193) & 2) != 0 )
  {
    v11 = a2;
    a2 = a1 + 64;
    if ( (unsigned int)sub_64A440(a1, a1 + 64) )
    {
      a1 = v11;
      sub_71B520(v11);
    }
  }
  sub_863FC0(a1, a2, a3, a4, a5);
  sub_866010(a1, a2, v7, v8, v9);
  dword_4F04C44 = a3[2];
  unk_4F04C2C = a3[1];
  *(_BYTE *)(*(_QWORD *)v5 + 81LL) |= 2u;
  sub_8CBAA0(v5);
  result = *a3;
  if ( (_DWORD)result )
    return sub_8D0B10();
  return result;
}
