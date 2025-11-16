// Function: sub_29AB500
// Address: 0x29ab500
//
void __fastcall sub_29AB500(__int64 a1, __int64 a2)
{
  _QWORD **v2; // rcx
  _QWORD *v3; // rbx
  _QWORD **v4; // r13
  _QWORD *v5; // r12
  unsigned __int64 *v6; // r15
  __int64 v7; // rax
  unsigned __int64 v8; // rdx
  _QWORD **v9; // [rsp+8h] [rbp-38h]

  v2 = *(_QWORD ***)(a1 + 88);
  v3 = *(_QWORD **)(a2 + 80);
  v9 = &v2[*(unsigned int *)(a1 + 96)];
  if ( v9 != v2 )
  {
    v4 = *(_QWORD ***)(a1 + 88);
    do
    {
      v5 = *v4++;
      sub_AA4A70(v5);
      v6 = (unsigned __int64 *)v3[1];
      sub_B2B790(a2 + 72, (__int64)v5);
      v7 = v5[3];
      v3 = v5 + 3;
      v8 = *v6;
      v5[4] = v6;
      v8 &= 0xFFFFFFFFFFFFFFF8LL;
      v5[3] = v8 | v7 & 7;
      *(_QWORD *)(v8 + 8) = v5 + 3;
      *v6 = (unsigned __int64)(v5 + 3) | *v6 & 7;
      sub_AA4C30((__int64)v5, *(_BYTE *)(a2 + 128));
    }
    while ( v9 != v4 );
  }
}
