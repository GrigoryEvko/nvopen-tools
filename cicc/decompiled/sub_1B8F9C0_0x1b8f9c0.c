// Function: sub_1B8F9C0
// Address: 0x1b8f9c0
//
__int64 __fastcall sub_1B8F9C0(_QWORD *a1, unsigned int *a2)
{
  __int64 v2; // rbx
  unsigned int v3; // r14d
  __int64 v4; // r12
  __int64 v5; // rdi
  __int64 v6; // r14

  v2 = a1[1];
  if ( *(_BYTE *)(v2 + 16) == 60 )
  {
    v3 = *a2;
    v4 = *(_QWORD *)(*a1 + 32LL);
    sub_1B8E090(**(__int64 ***)(v2 - 24), *a2);
    if ( *(_BYTE *)(*(_QWORD *)v2 + 8LL) )
    {
      if ( v3 != 1 )
        sub_16463B0(*(__int64 **)v2, v3);
    }
    v5 = *(_QWORD *)(v4 + 320);
    v6 = *(_QWORD *)(v2 - 24);
    if ( v6 == *(_QWORD *)(v5 + 64) )
      return sub_1BF2700(v5, v6);
    if ( !(unsigned __int8)sub_14A2CF0(*(_QWORD *)(v4 + 328)) )
    {
      v5 = *(_QWORD *)(v4 + 320);
      return sub_1BF2700(v5, v6);
    }
  }
  return 0;
}
