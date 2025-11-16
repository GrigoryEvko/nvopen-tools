// Function: sub_2018A30
// Address: 0x2018a30
//
__int64 __fastcall sub_2018A30(__int64 a1, __int64 a2, __int64 a3, __int64 a4, _QWORD *a5, _QWORD *a6)
{
  __int64 v6; // rdi
  __int64 v8; // [rsp+8h] [rbp-8h] BYREF

  *(_DWORD *)(a2 + 28) = -1;
  v6 = *(_QWORD *)(a1 + 32);
  v8 = a2;
  return sub_2018710(v6, &v8, a3, a4, a5, a6);
}
