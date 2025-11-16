// Function: sub_30B2870
// Address: 0x30b2870
//
__int64 __fastcall sub_30B2870(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // r12
  __int64 v3; // rax

  v1 = sub_22077B0(0x40u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 0;
    v3 = v1 + 56;
    *(_QWORD *)(v3 - 40) = 0;
    *(_QWORD *)(v3 - 32) = 0;
    *(_DWORD *)(v3 - 24) = 0;
    *(_QWORD *)(v2 + 40) = v3;
    *(_QWORD *)(v2 + 48) = 0;
    *(_DWORD *)(v2 + 56) = 4;
    *(_QWORD *)v2 = &unk_4A32368;
  }
  sub_30B2450(*(_QWORD *)(a1 + 8), v2);
  return v2;
}
