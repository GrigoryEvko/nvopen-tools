// Function: sub_87F360
// Address: 0x87f360
//
_QWORD *__fastcall sub_87F360(__int64 a1)
{
  __int64 v1; // rcx
  __int64 v2; // r8
  __int64 v3; // r9
  _QWORD *v4; // r12
  __int64 v5; // rdx

  v4 = sub_87EBB0(7u, *(_QWORD *)a1, (_QWORD *)(a1 + 48));
  *((_DWORD *)v4 + 10) = *(_DWORD *)(a1 + 40);
  v5 = *(_QWORD *)(a1 + 64);
  if ( (*(_BYTE *)(a1 + 81) & 0x10) != 0 )
  {
    sub_877E20((__int64)v4, 0, v5, v1, v2, v3);
    return v4;
  }
  else
  {
    if ( v5 )
      sub_877E90((__int64)v4, 0, v5);
    return v4;
  }
}
