// Function: sub_D33F60
// Address: 0xd33f60
//
__int64 __fastcall sub_D33F60(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // rax
  __int64 v7; // rbx
  __int64 v8; // r13
  char v9; // r14
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rsi
  unsigned int v13; // eax
  __int64 v15; // rax
  signed __int64 v16; // rcx
  __int64 v17; // rax
  __int64 v18; // rt2
  __int64 v19; // [rsp+10h] [rbp-30h]

  v6 = sub_D33D80(a1, a4, a3, a4, a5);
  if ( !*(_WORD *)(v6 + 24) )
  {
    v7 = v6;
    v8 = sub_AA4E30(**(_QWORD **)(a2 + 32));
    v9 = sub_AE5020(v8, a3);
    v10 = sub_9208B0(v8, a3);
    v11 = *(_QWORD *)(v7 + 32);
    v19 = v10;
    v12 = v10;
    v13 = *(_DWORD *)(v11 + 32);
    if ( v13 <= 0x40 )
    {
      if ( v13 )
      {
        v15 = (__int64)(*(_QWORD *)(v11 + 24) << (64 - (unsigned __int8)v13)) >> (64 - (unsigned __int8)v13);
        v16 = (((unsigned __int64)(v12 + 7) >> 3) + (1LL << v9) - 1) >> v9 << v9;
        v18 = v15 % v16;
        v17 = v15 / v16;
        if ( !v18 )
          return v17;
      }
      else
      {
        return 0;
      }
    }
  }
  return v19;
}
