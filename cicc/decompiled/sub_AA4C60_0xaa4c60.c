// Function: sub_AA4C60
// Address: 0xaa4c60
//
void __fastcall sub_AA4C60(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r15
  __int64 v4; // r14
  __int64 v7; // rcx
  __int64 v8; // rdx
  __int64 v9; // rdx
  __int64 v10; // rax
  char v11; // si

  v3 = a2 + 72;
  v4 = a1 + 24;
  if ( a3 )
  {
    sub_B2B790(v3, a1);
    v7 = *(_QWORD *)(a3 + 24);
    v8 = *(_QWORD *)(a1 + 24);
    *(_QWORD *)(a1 + 32) = a3 + 24;
    v7 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(a1 + 24) = v7 | v8 & 7;
    *(_QWORD *)(v7 + 8) = v4;
    *(_QWORD *)(a3 + 24) = v4 | *(_QWORD *)(a3 + 24) & 7LL;
    sub_AA4C30(a1, *(_BYTE *)(a2 + 128));
  }
  else
  {
    sub_B2B790(v3, a1);
    v9 = *(_QWORD *)(a2 + 72);
    v10 = *(_QWORD *)(a1 + 24);
    *(_QWORD *)(a1 + 32) = v3;
    v9 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(a1 + 24) = v9 | v10 & 7;
    *(_QWORD *)(v9 + 8) = v4;
    v11 = *(_BYTE *)(a2 + 128);
    *(_QWORD *)(a2 + 72) = v4 | *(_QWORD *)(a2 + 72) & 7LL;
    sub_AA4C30(a1, v11);
  }
  sub_AA4C30(a1, *(_BYTE *)(a2 + 128));
}
