// Function: sub_118DDB0
// Address: 0x118ddb0
//
__int64 __fastcall sub_118DDB0(__int64 a1, unsigned __int8 *a2, unsigned __int8 *a3)
{
  unsigned __int8 *v5; // rdi
  unsigned __int8 *v6; // rax
  __int64 v7; // r14
  unsigned __int8 v8; // al
  __int64 v9; // r12
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // rsi
  unsigned __int64 v14; // rax
  __int64 v15; // rbx
  __int64 v16; // rdx
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r13
  __int64 v20; // r13
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r9
  __int64 v25; // rax
  __int64 v26[5]; // [rsp+8h] [rbp-28h] BYREF

  v5 = **(unsigned __int8 ***)a1;
  if ( v5 == a2 && (*v5 <= 0x15u || *a3 > 0x15u) )
    return 0;
  v6 = sub_101E970(v5, (__int64)a2, (__int64)a3, (const __m128i *)(*(_QWORD *)(a1 + 8) + 96LL), 1, 0);
  v7 = (__int64)v6;
  if ( !v6 )
  {
LABEL_7:
    if ( **(unsigned __int8 ***)(a1 + 32) == a2
      && *a3 <= 0x15u
      && *a3 != 5
      && !(unsigned __int8)sub_AD6CA0((__int64)a3)
      && *a2 > 0x15u
      && sub_98EF80(
           a3,
           *(_QWORD *)(*(_QWORD *)(a1 + 8) + 128LL),
           *(_QWORD *)(a1 + 16),
           *(_QWORD *)(*(_QWORD *)(a1 + 8) + 80LL),
           0)
      && (unsigned __int8)sub_1188470(*(_QWORD *)(a1 + 8), **(_QWORD **)a1, (__int64)a2, (__int64)a3, 0) )
    {
      return *(_QWORD *)(a1 + 16);
    }
    return 0;
  }
  v8 = *v6;
  if ( v8 == 5
    || v8 > 0x15u
    || (unsigned __int8)sub_AD6CA0(v7)
    || !sub_98EF80(
          (unsigned __int8 *)v7,
          *(_QWORD *)(*(_QWORD *)(a1 + 8) + 128LL),
          *(_QWORD *)(a1 + 16),
          *(_QWORD *)(*(_QWORD *)(a1 + 8) + 80LL),
          0) )
  {
    if ( *a3 <= 0x15u && *a3 != 5 && !(unsigned __int8)sub_AD6CA0((__int64)a3) || (unsigned __int8 *)v7 == a3 )
    {
      if ( sub_98EF80(
             a3,
             *(_QWORD *)(*(_QWORD *)(a1 + 8) + 128LL),
             *(_QWORD *)(a1 + 16),
             *(_QWORD *)(*(_QWORD *)(a1 + 8) + 80LL),
             0) )
      {
        v9 = *(_QWORD *)(a1 + 16);
        v13 = *(_QWORD *)(a1 + 8);
        if ( (*(_BYTE *)(v9 + 7) & 0x40) != 0 )
          v14 = *(_QWORD *)(v9 - 8) + (-(__int64)(**(_BYTE **)(a1 + 24) == 0) & 0xFFFFFFFFFFFFFFE0LL) + 64;
        else
          v14 = v9 + 32 * (2LL - (**(_BYTE **)(a1 + 24) == 0) - (*(_DWORD *)(v9 + 4) & 0x7FFFFFF));
        v15 = *(_QWORD *)v14;
        if ( *(_QWORD *)v14 )
        {
          v16 = *(_QWORD *)(v14 + 8);
          **(_QWORD **)(v14 + 16) = v16;
          if ( v16 )
            *(_QWORD *)(v16 + 16) = *(_QWORD *)(v14 + 16);
        }
        *(_QWORD *)v14 = v7;
        v17 = *(_QWORD *)(v7 + 16);
        v18 = v7 + 16;
        *(_QWORD *)(v14 + 8) = v17;
        if ( v17 )
          *(_QWORD *)(v17 + 16) = v14 + 8;
        *(_QWORD *)(v14 + 16) = v18;
        *(_QWORD *)(v7 + 16) = v14;
        if ( *(_BYTE *)v15 > 0x1Cu )
        {
          v19 = *(_QWORD *)(v13 + 40);
          v26[0] = v15;
          v20 = v19 + 2096;
          sub_1187E30(v20, v26, v17, v18, v11, v12);
          v25 = *(_QWORD *)(v15 + 16);
          if ( v25 )
          {
            if ( !*(_QWORD *)(v25 + 8) )
            {
              v26[0] = *(_QWORD *)(v25 + 24);
              sub_1187E30(v20, v26, v21, v22, v23, v24);
            }
          }
        }
        return v9;
      }
      return 0;
    }
    goto LABEL_7;
  }
  return sub_F20660(*(_QWORD *)(a1 + 8), *(_QWORD *)(a1 + 16), 1 - ((unsigned int)(**(_BYTE **)(a1 + 24) == 0) - 1), v7);
}
