// Function: sub_3243610
// Address: 0x3243610
//
__int64 __fastcall sub_3243610(__int64 *a1, __int64 a2)
{
  unsigned __int64 *v2; // r13
  char v3; // cl
  __int64 v4; // rax

  if ( *(_QWORD *)a2 != *(_QWORD *)(a2 + 8) )
  {
    v2 = *(unsigned __int64 **)a2;
    *(_QWORD *)a2 = &v2[(unsigned int)sub_AF4160((unsigned __int64 **)a2)];
  }
  v3 = *((_BYTE *)a1 + 100);
  *((_BYTE *)a1 + 8) = 1;
  *((_BYTE *)a1 + 100) = v3 & 0xC0 | (8 * (v3 & 7) + 1);
  v4 = *a1;
  *((_WORD *)a1 + 50) |= 0x40u;
  return (*(__int64 (__fastcall **)(__int64 *))(v4 + 48))(a1);
}
