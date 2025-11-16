// Function: sub_1601A80
// Address: 0x1601a80
//
__int64 __fastcall sub_1601A80(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rax
  __int64 v5; // [rsp+0h] [rbp-30h] BYREF
  char v6; // [rsp+10h] [rbp-20h]

  v2 = *(_QWORD *)(*(_QWORD *)(a2 + 24 * (2LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF))) + 24LL);
  sub_15B1350((__int64)&v5, *(unsigned __int64 **)(v2 + 24), *(unsigned __int64 **)(v2 + 32));
  if ( v6 )
  {
    v3 = v5;
    *(_BYTE *)(a1 + 8) = 1;
    *(_QWORD *)a1 = v3;
  }
  else
  {
    sub_15B1130(a1, *(_QWORD *)(*(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF))) + 24LL));
  }
  return a1;
}
