// Function: sub_6E31E0
// Address: 0x6e31e0
//
__int64 __fastcall sub_6E31E0(const __m128i *a1, int a2, int a3, __int64 *a4)
{
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // rdi
  __int64 result; // rax

  v7 = sub_6E3060(a1);
  if ( a3 )
  {
    v8 = qword_4D03C50;
    v9 = *(_QWORD *)(qword_4D03C50 + 48LL);
    if ( v9 )
    {
      *(_QWORD *)(v7 + 32) = v9;
      v10 = *(_QWORD *)(qword_4F06BC0 + 32LL);
      *(_QWORD *)(v8 + 48) = 0;
      v11 = *(_QWORD *)(v7 + 32);
      qword_4F06BC0 = v10;
      sub_7347F0(v11);
    }
    *(_BYTE *)(v7 + 9) |= 1u;
  }
  sub_6E1850((__int64)a1);
  sub_6E1C20((_QWORD *)v7, a2, a4);
  result = qword_4D03C50;
  if ( (*(_BYTE *)(qword_4D03C50 + 20LL) & 4) != 0 )
    *(_BYTE *)(v7 + 9) |= 0x80u;
  return result;
}
