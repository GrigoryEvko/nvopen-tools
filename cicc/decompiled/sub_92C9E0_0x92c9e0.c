// Function: sub_92C9E0
// Address: 0x92c9e0
//
__int64 __fastcall sub_92C9E0(__int64 a1, __int64 a2, char a3, __int64 a4, unsigned __int8 a5, char a6, _DWORD *a7)
{
  __int64 v7; // rax
  __int64 v9[4]; // [rsp+8h] [rbp-20h] BYREF

  v9[1] = a1 + 48;
  v7 = *(_QWORD *)(a1 + 40);
  v9[0] = a1;
  v9[2] = v7;
  return sub_92BD50(v9, a2, a3, a4, a5, a6, a7);
}
