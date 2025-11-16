// Function: sub_6F6DD0
// Address: 0x6f6dd0
//
__int64 __fastcall sub_6F6DD0(__int64 a1)
{
  _QWORD *v1; // r13
  _QWORD *v2; // r14
  __int64 v3; // r12
  __m128i v5[8]; // [rsp+0h] [rbp-180h] BYREF
  __int64 v6; // [rsp+80h] [rbp-100h]

  v1 = (_QWORD *)sub_6E1A20(a1);
  v2 = (_QWORD *)sub_6E1A60(a1);
  v3 = sub_726700(25);
  *(_QWORD *)(v3 + 56) = sub_6F6D20(*(_QWORD *)(a1 + 24), (_DWORD *)1);
  *(_QWORD *)v3 = *(_QWORD *)&dword_4D03B80;
  if ( *(_QWORD *)(a1 + 16) )
    *(_BYTE *)(v3 + 26) |= 4u;
  *(_QWORD *)(v3 + 28) = *v1;
  *(_QWORD *)(v3 + 36) = *v1;
  *(_QWORD *)(v3 + 44) = *v2;
  if ( qword_4D03C50 && (*(_BYTE *)(qword_4D03C50 + 19LL) & 2) != 0 )
  {
    sub_6E9FE0(a1, v5);
    v6 = *(_QWORD *)(a1 + 16);
    sub_6E39C0(v5, v3);
  }
  return v3;
}
