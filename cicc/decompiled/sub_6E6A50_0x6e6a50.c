// Function: sub_6E6A50
// Address: 0x6e6a50
//
__int64 __fastcall sub_6E6A50(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  char i; // dl
  __int64 result; // rax
  __int64 v5; // rax

  v2 = *(_QWORD *)(a1 + 128);
  for ( i = *(_BYTE *)(v2 + 140); i == 12; i = *(_BYTE *)(v2 + 140) )
    v2 = *(_QWORD *)(v2 + 160);
  if ( i )
  {
    sub_6E2E50(2, a2);
    sub_72A510(a1, a2 + 144);
    if ( *(_BYTE *)(a2 + 317) == 12 && *(_BYTE *)(a2 + 320) == 1 && (*(_BYTE *)(a2 + 321) & 0x10) != 0 )
    {
      v5 = sub_72E9A0(a1);
      *(_BYTE *)(a2 + 321) &= ~0x10u;
      *(_QWORD *)(a2 + 328) = v5;
    }
    *(_QWORD *)a2 = *(_QWORD *)(a1 + 128);
  }
  else
  {
    sub_6E6260((_QWORD *)a2);
  }
  *(_BYTE *)(a2 + 17) = (*(_BYTE *)(a1 + 173) != 2) + 1;
  *(_QWORD *)(a2 + 68) = *(_QWORD *)&dword_4F063F8;
  result = qword_4F063F0;
  *(_QWORD *)(a2 + 76) = qword_4F063F0;
  return result;
}
