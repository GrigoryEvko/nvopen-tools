// Function: sub_1CED350
// Address: 0x1ced350
//
__int64 __fastcall sub_1CED350(__int64 a1)
{
  __int64 v1; // rax
  unsigned __int64 v2; // rax
  __int64 v3; // r12
  __int64 v4; // rax
  __int64 v5; // rbx
  __int64 v6; // rax
  __int64 v7; // r12
  __int64 v8; // rax
  __int64 v9; // rbx
  __int64 v10; // r15
  _QWORD *v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // r14
  _BYTE *v16; // r13
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rdx
  __int64 result; // rax
  _BYTE *v21; // rdi
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rdx

  v1 = sub_13FCB50(a1);
  if ( !v1 )
    return 0xFFFFFFFFLL;
  v2 = sub_157EBA0(v1);
  v3 = v2;
  if ( *(_BYTE *)(v2 + 16) != 26 )
    return 0xFFFFFFFFLL;
  if ( !*(_QWORD *)(v2 + 48) && *(__int16 *)(v2 + 18) >= 0 )
    goto LABEL_23;
  v4 = sub_1625940(v2, "pragma", 6u);
  v5 = v4;
  if ( !v4
    || *(_DWORD *)(v4 + 8) != 2
    || (v21 = *(_BYTE **)(v4 - 16), *v21)
    || (v22 = sub_161E970((__int64)v21), v23 != 6)
    || *(_DWORD *)v22 != 1869770357
    || *(_WORD *)(v22 + 4) != 27756 )
  {
    if ( *(_QWORD *)(v3 + 48) )
    {
LABEL_7:
      v6 = sub_1625940(v3, "llvm.loop", 9u);
      v7 = v6;
      if ( v6 )
      {
        v8 = *(unsigned int *)(v6 + 8);
        if ( (unsigned int)v8 > 1 )
        {
          v9 = v8;
          v10 = 1;
          while ( 1 )
          {
            v15 = *(_QWORD *)(v7 + 8 * (v10 - v8));
            if ( (unsigned __int8)(*(_BYTE *)v15 - 4) <= 0x1Eu )
            {
              v16 = *(_BYTE **)(v15 - 8LL * *(unsigned int *)(v15 + 8));
              if ( !*v16 )
              {
                v17 = sub_161E970(*(_QWORD *)(v15 - 8LL * *(unsigned int *)(v15 + 8)));
                if ( v18 == 22
                  && !(*(_QWORD *)v17 ^ 0x6F6F6C2E6D766C6CLL | *(_QWORD *)(v17 + 8) ^ 0x6C6C6F726E752E70LL)
                  && *(_DWORD *)(v17 + 16) == 1970234158
                  && *(_WORD *)(v17 + 20) == 29806 )
                {
                  v19 = *(_QWORD *)(*(_QWORD *)(v15 + 8 * (1LL - *(unsigned int *)(v15 + 8))) + 136LL);
                  result = *(_QWORD *)(v19 + 24);
                  if ( *(_DWORD *)(v19 + 32) > 0x40u )
                    return *(_QWORD *)result;
                  return result;
                }
                v11 = (_QWORD *)sub_161E970((__int64)v16);
                if ( v12 == 24
                  && !(*v11 ^ 0x6F6F6C2E6D766C6CLL | v11[1] ^ 0x6C6C6F726E752E70LL)
                  && v11[2] == 0x656C62617369642ELL )
                {
                  return 1;
                }
                v13 = sub_161E970((__int64)v16);
                if ( v14 == 21
                  && !(*(_QWORD *)v13 ^ 0x6F6F6C2E6D766C6CLL | *(_QWORD *)(v13 + 8) ^ 0x6C6C6F726E752E70LL)
                  && *(_DWORD *)(v13 + 16) == 1819633198
                  && *(_BYTE *)(v13 + 20) == 108 )
                {
                  return 0x7FFFFFFF;
                }
              }
            }
            if ( v9 == ++v10 )
              return 0xFFFFFFFFLL;
            v8 = *(unsigned int *)(v7 + 8);
          }
        }
      }
      return 0xFFFFFFFFLL;
    }
LABEL_23:
    if ( *(__int16 *)(v3 + 18) >= 0 )
      return 0xFFFFFFFFLL;
    goto LABEL_7;
  }
  v24 = *(_QWORD *)(*(_QWORD *)(v5 + 8 * (1LL - *(unsigned int *)(v5 + 8))) + 136LL);
  result = *(_QWORD *)(v24 + 24);
  if ( *(_DWORD *)(v24 + 32) > 0x40u )
    return *(_QWORD *)result;
  return result;
}
