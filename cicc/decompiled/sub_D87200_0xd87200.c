// Function: sub_D87200
// Address: 0xd87200
//
__int64 __fastcall sub_D87200(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // [rsp+0h] [rbp-40h] BYREF
  int v6; // [rsp+8h] [rbp-38h]
  __int64 v7; // [rsp+10h] [rbp-30h]
  int v8; // [rsp+18h] [rbp-28h]

  if ( (unsigned int)sub_ABDC10(a2, a3) == 3 )
  {
    sub_AB4F10((__int64)&v5, a2, a3);
    *(_DWORD *)(a1 + 8) = v6;
    *(_QWORD *)a1 = v5;
    *(_DWORD *)(a1 + 24) = v8;
    *(_QWORD *)(a1 + 16) = v7;
  }
  else
  {
    sub_AADB10(a1, *(_DWORD *)(a2 + 8), 1);
  }
  return a1;
}
