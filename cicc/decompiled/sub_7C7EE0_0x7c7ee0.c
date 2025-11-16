// Function: sub_7C7EE0
// Address: 0x7c7ee0
//
_QWORD *__fastcall sub_7C7EE0(__int64 a1, int a2, _DWORD *a3)
{
  _QWORD *v4; // r14
  __int64 v6[5]; // [rsp+8h] [rbp-28h] BYREF

  v6[0] = -1;
  ++*(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 632);
  v4 = sub_7C6D90(a1, a2, a3, 0, v6);
  if ( word_4F06418[0] != 44 )
    sub_7BC0A0(a3);
  --*(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 632);
  return v4;
}
