// Function: sub_E22500
// Address: 0xe22500
//
unsigned __int64 __fastcall sub_E22500(__int64 a1, __int64 *a2, char a3)
{
  size_t v3; // rax
  __int64 v4; // rdx
  __int64 v5; // r13
  _QWORD *v6; // rdx
  size_t v7; // r12
  unsigned __int64 result; // rax
  unsigned __int64 *v9; // rax
  unsigned __int64 *v10; // r14
  unsigned __int64 v11; // rdx

  v3 = sub_E22480(a1, a2, a3);
  if ( *(_BYTE *)(a1 + 8) )
    return 0;
  v5 = v4;
  v6 = *(_QWORD **)(a1 + 16);
  v7 = v3;
  result = (*v6 + v6[1] + 7LL) & 0xFFFFFFFFFFFFFFF8LL;
  v6[1] = result - *v6 + 40;
  if ( *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) > *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL) )
  {
    v9 = (unsigned __int64 *)sub_22077B0(32);
    v10 = v9;
    if ( v9 )
    {
      *v9 = 0;
      v9[1] = 0;
      v9[2] = 0;
      v9[3] = 0;
    }
    result = sub_2207820(4096);
    v11 = *(_QWORD *)(a1 + 16);
    v10[2] = 4096;
    *v10 = result;
    v10[3] = v11;
    *(_QWORD *)(a1 + 16) = v10;
    v10[1] = 40;
  }
  if ( !result )
  {
    MEMORY[0x18] = v7;
    MEMORY[0x20] = v5;
    BUG();
  }
  *(_QWORD *)(result + 24) = 0;
  *(_QWORD *)(result + 32) = 0;
  *(_DWORD *)(result + 8) = 5;
  *(_QWORD *)(result + 16) = 0;
  *(_QWORD *)result = &unk_49E0F88;
  *(_QWORD *)(result + 24) = v7;
  *(_QWORD *)(result + 32) = v5;
  return result;
}
