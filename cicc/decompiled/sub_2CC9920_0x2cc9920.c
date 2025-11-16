// Function: sub_2CC9920
// Address: 0x2cc9920
//
bool __fastcall sub_2CC9920(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 v5; // rdx
  unsigned __int64 v6; // r8
  bool result; // al
  __int64 v8; // r14
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // r8
  unsigned __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rdx
  unsigned __int64 v15; // r15
  unsigned int v16; // r12d
  __int64 v17; // rax
  __int64 v18; // rdx
  unsigned __int64 v19; // rax
  char v20; // [rsp+Ch] [rbp-44h]
  unsigned __int64 v21; // [rsp+10h] [rbp-40h] BYREF
  __int64 v22; // [rsp+18h] [rbp-38h]

  v4 = sub_9208B0(a3, *(_QWORD *)(a2 + 8));
  v22 = v5;
  v21 = (unsigned __int64)(v4 + 7) >> 3;
  v6 = sub_CA1930(&v21);
  result = 1;
  if ( v6 < (unsigned int)qword_5013708 )
  {
    v8 = *(_QWORD *)(a2 + 8);
    if ( *(_BYTE *)(v8 + 8) == 15
      && *(_DWORD *)(v8 + 12)
      && (v9 = sub_9208B0(a3, *(_QWORD *)(a2 + 8)),
          v22 = v10,
          v21 = (unsigned __int64)(v9 + 7) >> 3,
          v11 = sub_CA1930(&v21),
          _BitScanReverse64(&v12, 1LL << (*(_WORD *)(a2 + 2) >> 1)),
          v20 = 63 - (v12 ^ 0x3F),
          (v11 & ~(-1LL << v20)) == 0) )
    {
      v13 = sub_9208B0(a3, **(_QWORD **)(v8 + 16));
      v22 = v14;
      v21 = (unsigned __int64)(v13 + 7) >> 3;
      v15 = sub_CA1930(&v21);
      if ( *(_DWORD *)(v8 + 12) > 1u )
      {
        v16 = 1;
        do
        {
          v17 = sub_9208B0(a3, *(_QWORD *)(*(_QWORD *)(v8 + 16) + 8LL * v16));
          v22 = v18;
          v21 = (unsigned __int64)(v17 + 7) >> 3;
          v19 = sub_CA1930(&v21);
          if ( v15 > v19 )
            v15 = v19;
          ++v16;
        }
        while ( *(_DWORD *)(v8 + 12) > v16 );
      }
      return 1LL << v20 > v15;
    }
    else
    {
      return 0;
    }
  }
  return result;
}
