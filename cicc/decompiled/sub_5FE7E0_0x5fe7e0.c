// Function: sub_5FE7E0
// Address: 0x5fe7e0
//
__int64 __fastcall sub_5FE7E0(__int64 *a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  _BYTE v5[112]; // [rsp+0h] [rbp-2E0h] BYREF
  _QWORD v6[70]; // [rsp+70h] [rbp-270h] BYREF
  char v7; // [rsp+2A0h] [rbp-40h]

  sub_5E4C60((__int64)v6, (_QWORD *)(*a1 + 64));
  v7 |= 8u;
  sub_87E3B0(v5);
  sub_5FE480(a1, (__int64)v6, (__int64)v5, 0);
  v2 = v6[0];
  if ( *(_DWORD *)(a2 + 28) )
  {
    if ( dword_4D04464 )
    {
      v3 = *(_QWORD *)(v6[0] + 88LL);
      *(_BYTE *)(v6[0] + 81LL) |= 2u;
      *(_BYTE *)(v3 + 206) |= 0x10u;
      *(_BYTE *)(*(_QWORD *)(v2 + 88) + 193LL) |= 0x20u;
    }
  }
  else if ( *(_DWORD *)&word_4D04898 && !*(_DWORD *)(a2 + 60) && unk_4D04880 )
  {
    *(_BYTE *)(*(_QWORD *)(v6[0] + 88LL) + 193LL) |= 2u;
  }
  return *(_QWORD *)(v2 + 88);
}
