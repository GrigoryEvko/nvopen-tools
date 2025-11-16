// Function: sub_390FEA0
// Address: 0x390fea0
//
__int64 __fastcall sub_390FEA0(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rax
  int v3; // r8d
  int v4; // r9d
  __int64 v5; // r12
  __int64 v6; // rax
  unsigned int v7; // edx

  result = *(_QWORD *)(a1 + 56);
  if ( !result )
  {
    v2 = sub_22077B0(0xE0u);
    v5 = v2;
    if ( v2 )
    {
      sub_38CF760(v2, 1, 0, 0);
      *(_QWORD *)(a1 + 56) = v5;
      *(_WORD *)(v5 + 48) = 0;
      *(_QWORD *)(v5 + 64) = v5 + 80;
      *(_QWORD *)(v5 + 72) = 0x2000000000LL;
      *(_QWORD *)(v5 + 56) = 0;
      *(_QWORD *)(v5 + 112) = v5 + 128;
      *(_QWORD *)(v5 + 120) = 0x400000000LL;
      v6 = 0;
    }
    else
    {
      v6 = MEMORY[0x48];
      v7 = MEMORY[0x4C];
      *(_QWORD *)(a1 + 56) = 0;
      if ( v7 <= (unsigned int)v6 )
      {
        sub_16CD150(64, (const void *)0x50, 0, 1, v3, v4);
        v6 = MEMORY[0x48];
      }
    }
    *(_BYTE *)(*(_QWORD *)(v5 + 64) + v6) = 0;
    result = *(_QWORD *)(a1 + 56);
    ++*(_DWORD *)(v5 + 72);
  }
  return result;
}
