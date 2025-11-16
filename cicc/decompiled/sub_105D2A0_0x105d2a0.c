// Function: sub_105D2A0
// Address: 0x105d2a0
//
__int64 *__fastcall sub_105D2A0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rbx
  __int64 v7; // r15
  __int64 v8; // rax
  __int64 *v9; // rdx

  v6 = sub_BC1CD0(a4, &unk_4F81450, a3);
  v7 = sub_BC1CD0(a4, &unk_4F89C30, a3) + 8;
  v8 = sub_BC1CD0(a4, &unk_4F92388, a3);
  sub_105CFE0(a1, v6 + 8, (__int64 *)(v8 + 8), v7);
  if ( (unsigned __int8)sub_DF9710(v7) )
  {
    sub_1058900(*a1, a3, v9);
    sub_105BE30(*a1);
  }
  return a1;
}
