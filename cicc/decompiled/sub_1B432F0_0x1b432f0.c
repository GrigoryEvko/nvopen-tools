// Function: sub_1B432F0
// Address: 0x1b432f0
//
__int64 __fastcall sub_1B432F0(__int64 a1)
{
  __int64 v2; // rax
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 v5; // r14
  unsigned int v6; // r15d
  __int64 v7; // rdx
  unsigned int v8; // r13d
  __int64 v9; // rcx
  int v10; // eax
  _BYTE *v11; // rdi
  __int64 v12; // rax
  unsigned __int64 v13; // rdx
  _BYTE *v14; // rdi
  __int64 v15; // rax
  unsigned __int64 v16; // rdx
  _BYTE *v18; // rdi
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // [rsp+8h] [rbp-38h]

  v2 = *(_QWORD *)(a1 - 72);
  if ( *(_BYTE *)(v2 + 16) == 78 )
  {
    v21 = *(_QWORD *)(v2 - 24);
    if ( !*(_BYTE *)(v21 + 16) && (*(_BYTE *)(v21 + 33) & 0x20) != 0 && *(_DWORD *)(v21 + 36) == 3 )
      return 1;
  }
  if ( *(_QWORD *)(a1 + 48) || *(__int16 *)(a1 + 18) < 0 )
  {
    v3 = sub_1625940(a1, "pragma", 6u);
    if ( v3 )
    {
      if ( *(_DWORD *)(v3 + 8) == 2 )
      {
        v18 = *(_BYTE **)(v3 - 16);
        if ( !*v18 )
        {
          v19 = sub_161E970((__int64)v18);
          if ( v20 == 6 && *(_DWORD *)v19 == 1869770357 && *(_WORD *)(v19 + 4) == 27756 )
            return 1;
        }
      }
    }
    if ( *(_QWORD *)(a1 + 48) )
      goto LABEL_39;
  }
  if ( *(__int16 *)(a1 + 18) < 0 )
  {
LABEL_39:
    v4 = sub_1625940(a1, "llvm.loop", 9u);
    v5 = v4;
    if ( v4 )
    {
      v6 = *(_DWORD *)(v4 + 8);
      if ( v6 != 1 )
      {
        v7 = v6;
        v8 = 1;
        while ( 1 )
        {
          v9 = *(_QWORD *)(v5 + 8 * (v8 - v7));
          if ( (unsigned __int8)(*(_BYTE *)v9 - 4) > 0x1Eu )
            goto LABEL_15;
          v10 = *(_DWORD *)(v9 + 8);
          if ( v10 == 1 )
          {
            v14 = *(_BYTE **)(v9 - 8);
            v22 = *(_QWORD *)(v5 + 8 * (v8 - v7));
            if ( *v14 )
              goto LABEL_15;
            v15 = sub_161E970((__int64)v14);
            v9 = v22;
            if ( v16 > 0x14
              && !(*(_QWORD *)v15 ^ 0x6F6F6C2E6D766C6CLL | *(_QWORD *)(v15 + 8) ^ 0x6C6C6F726E752E70LL)
              && *(_DWORD *)(v15 + 16) == 1819633198
              && *(_BYTE *)(v15 + 20) == 108 )
            {
              return 1;
            }
            v10 = *(_DWORD *)(v22 + 8);
          }
          if ( v10 == 2 )
          {
            v11 = *(_BYTE **)(v9 - 16);
            if ( !*v11 )
            {
              v12 = sub_161E970((__int64)v11);
              if ( v13 > 0x10
                && !(*(_QWORD *)v12 ^ 0x6F6F6C2E6D766C6CLL | *(_QWORD *)(v12 + 8) ^ 0x6C6C6F726E752E70LL)
                && *(_BYTE *)(v12 + 16) == 46 )
              {
                return 1;
              }
            }
          }
LABEL_15:
          if ( v6 == ++v8 )
            return 0;
          v7 = *(unsigned int *)(v5 + 8);
        }
      }
    }
  }
  return 0;
}
