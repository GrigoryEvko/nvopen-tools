// Function: sub_2BF0CC0
// Address: 0x2bf0cc0
//
__int64 __fastcall sub_2BF0CC0(__int64 a1, __int64 a2)
{
  __int64 v2; // r15
  unsigned __int64 v3; // rax
  unsigned __int64 v4; // r12
  __int64 v5; // rbx
  __int64 v6; // rax
  __int64 v7; // r12
  __int64 v8; // rax
  __int64 v9; // rsi
  __int64 v10; // rcx
  unsigned __int64 i; // [rsp+8h] [rbp-38h]

  v2 = sub_2BF0C30(a1, a2);
  v3 = *(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v3 == a2 + 48 )
  {
    v4 = 0;
  }
  else
  {
    if ( !v3 )
      BUG();
    v4 = v3 - 24;
    if ( (unsigned int)*(unsigned __int8 *)(v3 - 24) - 30 >= 0xB )
      v4 = 0;
  }
  v5 = *(_QWORD *)(a2 + 56);
  for ( i = v4 + 24; i != v5; v5 = *(_QWORD *)(v5 + 8) )
  {
    v6 = 0;
    if ( v5 )
      v6 = v5 - 24;
    v7 = v6;
    v8 = sub_22077B0(0x68u);
    if ( v8 )
    {
      *(_QWORD *)(v8 + 24) = 0;
      *(_QWORD *)(v8 + 48) = v8 + 64;
      *(_QWORD *)(v8 + 32) = 0;
      *(_BYTE *)(v8 + 8) = 3;
      *(_QWORD *)(v8 + 16) = 0;
      *(_QWORD *)(v8 + 56) = 0x200000000LL;
      *(_QWORD *)(v8 + 88) = 0;
      *(_QWORD *)(v8 + 40) = &unk_4A23C60;
      *(_QWORD *)(v8 + 96) = v7;
      *(_QWORD *)v8 = &unk_4A23C10;
      v9 = 0;
    }
    else
    {
      v9 = MEMORY[0x18] & 7;
    }
    *(_QWORD *)(v8 + 80) = v2;
    v10 = *(_QWORD *)(v2 + 112);
    *(_QWORD *)(v8 + 32) = v2 + 112;
    v10 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v8 + 24) = v10 | v9;
    *(_QWORD *)(v10 + 8) = v8 + 24;
    *(_QWORD *)(v2 + 112) = *(_QWORD *)(v2 + 112) & 7LL | (v8 + 24);
  }
  return v2;
}
