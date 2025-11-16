// Function: sub_E22610
// Address: 0xe22610
//
unsigned __int64 __fastcall sub_E22610(__int64 a1, __int64 *a2)
{
  __int64 v2; // rdx
  _QWORD *v3; // rax
  unsigned __int64 v4; // r12
  __int64 v6; // rdx
  __int64 *v7; // rax
  __int64 *v8; // rbx
  __int64 v9; // rax
  __int64 v10; // rax

  v3 = *(_QWORD **)(a1 + 16);
  v4 = (*v3 + v3[1] + 7LL) & 0xFFFFFFFFFFFFFFF8LL;
  v3[1] = v4 - *v3 + 40;
  if ( *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) > *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL) )
  {
    v7 = (__int64 *)sub_22077B0(32);
    v8 = v7;
    if ( v7 )
    {
      *v7 = 0;
      v7[1] = 0;
      v7[2] = 0;
      v7[3] = 0;
    }
    v9 = sub_2207820(4096);
    v8[2] = 4096;
    *v8 = v9;
    v4 = v9;
    v10 = *(_QWORD *)(a1 + 16);
    v8[1] = 40;
    v8[3] = v10;
    *(_QWORD *)(a1 + 16) = v8;
  }
  if ( !v4 )
  {
    MEMORY[0x18] = sub_E22480(a1, a2, 0);
    MEMORY[0x20] = v2;
    BUG();
  }
  *(_DWORD *)(v4 + 8) = 12;
  *(_QWORD *)(v4 + 16) = 0;
  *(_QWORD *)v4 = &unk_49E1050;
  *(_QWORD *)(v4 + 24) = 0;
  *(_QWORD *)(v4 + 32) = 0;
  *(_QWORD *)(v4 + 24) = sub_E22480(a1, a2, 0);
  *(_QWORD *)(v4 + 32) = v6;
  return v4;
}
