// Function: sub_28CF390
// Address: 0x28cf390
//
__int64 __fastcall sub_28CF390(__int64 a1, __int64 a2, __int64 a3)
{
  int v4; // r14d
  __int64 v5; // rax
  int v6; // r14d
  __int64 v7; // r12
  __int64 v8; // r13
  __int64 v9; // r14
  unsigned __int8 *v10; // rbx

  v4 = *(_DWORD *)(a2 + 4);
  v5 = sub_A777F0(0x40u, (__int64 *)(a1 + 72));
  v6 = v4 & 0x7FFFFFF;
  v7 = v5;
  if ( v5 )
  {
    *(_DWORD *)(v5 + 32) = v6;
    *(_QWORD *)(v5 + 8) = 0xFFFFFFFD0000000ALL;
    *(_QWORD *)(v5 + 16) = 0;
    *(_QWORD *)(v5 + 48) = a3;
    *(_QWORD *)(v5 + 24) = 0;
    *(_DWORD *)(v5 + 36) = 0;
    *(_QWORD *)(v5 + 40) = 0;
    *(_QWORD *)v5 = &unk_4A21A68;
    *(_QWORD *)(v5 + 56) = a2;
  }
  sub_28CF1F0(a1, (__int64 *)a2, v5);
  if ( sub_B46D50((unsigned __int8 *)a2) )
  {
    v8 = *(_QWORD *)(v7 + 24);
    v9 = *(_QWORD *)(v8 + 8);
    v10 = *(unsigned __int8 **)v8;
    if ( sub_28C8D50(a1, *(unsigned __int8 **)v8, v9) )
    {
      *(_QWORD *)v8 = v9;
      *(_QWORD *)(v8 + 8) = v10;
    }
  }
  return v7;
}
