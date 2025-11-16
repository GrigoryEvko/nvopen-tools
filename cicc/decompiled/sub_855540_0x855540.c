// Function: sub_855540
// Address: 0x855540
//
__int64 __fastcall sub_855540(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rdi
  __int64 v7; // rax
  __int64 result; // rax
  __int64 v9; // r12
  __int64 v10; // rax

  v6 = qword_4F5FCD0;
  v7 = unk_4D03CD8 + 1LL;
  if ( unk_4D03CD8 + 1LL == qword_4F5FCC8 )
  {
    v9 = unk_4D03CD8 + 31LL;
    v10 = sub_822C60((void *)qword_4F5FCD0, 12 * (unk_4D03CD8 + 31LL) - 360, 12 * (unk_4D03CD8 + 31LL), a4, a5, a6);
    qword_4F5FCC8 = v9;
    qword_4F5FCD0 = v10;
    v6 = v10;
    v7 = unk_4D03CD8 + 1LL;
  }
  unk_4D03CD8 = v7;
  *(_QWORD *)(v6 + 12 * v7) = qword_4F5FCC0;
  result = 3LL * unk_4D03CD8;
  *(_DWORD *)(v6 + 12LL * unk_4D03CD8 + 8) = 0;
  return result;
}
