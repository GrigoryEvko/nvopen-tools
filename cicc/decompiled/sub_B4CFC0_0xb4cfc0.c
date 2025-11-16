// Function: sub_B4CFC0
// Address: 0xb4cfc0
//
__int64 __fastcall sub_B4CFC0(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v4; // rax
  char v5; // dl
  __int64 v6; // [rsp+10h] [rbp-30h] BYREF
  char v7; // [rsp+18h] [rbp-28h]
  char v8; // [rsp+20h] [rbp-20h]

  sub_B4CED0((__int64)&v6, a2, a3);
  if ( v8 && (v4 = sub_B48740(v6, 8), v5) )
  {
    *(_BYTE *)(a1 + 16) = 1;
    *(_QWORD *)a1 = v4;
    *(_BYTE *)(a1 + 8) = v7;
    return a1;
  }
  else
  {
    *(_BYTE *)(a1 + 16) = 0;
    return a1;
  }
}
