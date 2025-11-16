// Function: sub_876D90
// Address: 0x876d90
//
_BOOL8 __fastcall sub_876D90(__int64 a1, __int64 a2, FILE *a3, unsigned int a4, _DWORD *a5)
{
  __int64 v5; // rax
  __int64 v6; // rbx

  for ( ; *(_BYTE *)(a1 + 140) == 12; a1 = *(_QWORD *)(a1 + 160) )
    ;
  v5 = *(_QWORD *)(*(_QWORD *)a1 + 96LL);
  v6 = *(_QWORD *)(v5 + 16);
  if ( v6 )
    sub_8769C0(*(_QWORD *)(v5 + 16), a3, a2, 0, 1, 1, a4, 0, a5);
  return v6 != 0;
}
