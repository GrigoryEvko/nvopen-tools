// Function: sub_145D1F0
// Address: 0x145d1f0
//
__int64 __fastcall sub_145D1F0(__int64 a1, __int64 a2)
{
  char v2; // al
  __int64 v3; // r13
  __int64 v4; // rax
  __int64 v5; // rax

  v2 = *(_BYTE *)(a2 + 16);
  if ( v2 == 55 )
  {
    v3 = **(_QWORD **)(a2 - 48);
LABEL_3:
    v4 = sub_1646BA0(v3, 0);
    v5 = sub_1456E10(a1, v4);
    return sub_145D050(a1, v5, v3);
  }
  if ( v2 == 54 )
  {
    v3 = *(_QWORD *)a2;
    goto LABEL_3;
  }
  return 0;
}
