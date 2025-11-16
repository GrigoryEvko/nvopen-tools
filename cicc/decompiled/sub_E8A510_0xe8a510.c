// Function: sub_E8A510
// Address: 0xe8a510
//
__int64 __fastcall sub_E8A510(__int64 a1, __int64 a2)
{
  void *v2; // r13
  void *v3; // rax
  void *v5; // rax
  __int64 v6; // [rsp+10h] [rbp-30h]

  v2 = *(void **)a1;
  if ( *(_QWORD *)a1
    || (*(_BYTE *)(a1 + 9) & 0x70) == 0x20
    && *(char *)(a1 + 8) >= 0
    && (*(_BYTE *)(a1 + 8) |= 8u, v5 = sub_E807D0(*(_QWORD *)(a1 + 24)), *(_QWORD *)a1 = v5, (v2 = v5) != 0) )
  {
    v3 = *(void **)a2;
    if ( !*(_QWORD *)a2 )
    {
      if ( (*(_BYTE *)(a2 + 9) & 0x70) != 0x20 || *(char *)(a2 + 8) < 0 )
        return v6;
      *(_BYTE *)(a2 + 8) |= 8u;
      v3 = sub_E807D0(*(_QWORD *)(a2 + 24));
      *(_QWORD *)a2 = v3;
    }
    if ( v3 == v2 && (*(_BYTE *)(a1 + 9) & 0x70) != 0x20 && (*(_BYTE *)(a2 + 9) & 0x70) != 0x20 )
      return *(_QWORD *)(a1 + 24) - *(_QWORD *)(a2 + 24);
  }
  return v6;
}
