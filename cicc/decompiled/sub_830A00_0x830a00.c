// Function: sub_830A00
// Address: 0x830a00
//
_QWORD *__fastcall sub_830A00(int a1)
{
  _QWORD *v1; // rdi
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // r9
  __int64 v8; // rax
  __m128i v9[23]; // [rsp+0h] [rbp-170h] BYREF

  v1 = *(_QWORD **)(qword_4F04C68[0] + 776LL * a1 + 184);
  v2 = **(_QWORD **)(*(_QWORD *)(v1[4] + 152LL) + 168LL);
  if ( v2 && (*(_BYTE *)(v2 + 35) & 1) != 0 )
  {
    sub_6F8E70(v1[5], &dword_4F063F8, &qword_4F063F0, v9, 0);
    sub_6FF5E0((__int64)v9, &dword_4F063F8);
    return (_QWORD *)sub_6F6F40(v9, 0, v3, v4, v5, v6);
  }
  else
  {
    v8 = sub_8309B0(v1);
    return sub_73E830(v8);
  }
}
