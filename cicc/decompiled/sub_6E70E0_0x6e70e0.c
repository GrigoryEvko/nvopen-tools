// Function: sub_6E70E0
// Address: 0x6e70e0
//
__int64 __fastcall sub_6E70E0(__int64 *a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // rax

  if ( *((_BYTE *)a1 + 24) )
  {
    sub_6E2E50(1, a2);
    v3 = *a1;
    *(_BYTE *)(a2 + 17) = 2;
    *(_QWORD *)(a2 + 144) = a1;
    *(_QWORD *)a2 = v3;
  }
  else
  {
    sub_6E6260((_QWORD *)a2);
  }
  *(_QWORD *)(a2 + 68) = *(_QWORD *)&dword_4F063F8;
  result = qword_4F063F0;
  *(_QWORD *)(a2 + 76) = qword_4F063F0;
  return result;
}
