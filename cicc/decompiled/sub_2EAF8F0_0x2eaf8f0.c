// Function: sub_2EAF8F0
// Address: 0x2eaf8f0
//
__int64 (__fastcall *__fastcall sub_2EAF8F0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4))(_QWORD *, _QWORD *, __int64)
{
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v10; // [rsp+8h] [rbp-A8h]
  _QWORD v11[20]; // [rsp+10h] [rbp-A0h] BYREF

  v6 = *(_QWORD *)(a1 + 16);
  if ( v6 )
  {
    v7 = *(_QWORD *)(v6 + 24);
    if ( v7 )
    {
      v8 = *(_QWORD *)(v7 + 32);
      if ( v8 )
        a4 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(v8 + 16) + 200LL))(*(_QWORD *)(v8 + 16));
    }
  }
  sub_A558A0((__int64)v11, 0, 1);
  BYTE4(v10) = 0;
  sub_2EAE5A0(a1, a2, (__int64)v11, a3, v10, 0, 1, 1, 0, a4);
  return sub_A55520(v11, a2);
}
