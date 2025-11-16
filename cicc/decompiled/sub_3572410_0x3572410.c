// Function: sub_3572410
// Address: 0x3572410
//
void __fastcall sub_3572410(__int64 *a1, __int64 a2)
{
  __int64 v4; // r13
  int v5; // ecx
  __int64 v6; // rsi
  unsigned int v7; // eax
  unsigned int v8; // edi
  __int64 v9; // rax
  int v10; // r14d
  __int64 v11; // rdx
  unsigned __int64 v12; // rdx
  __int64 v13; // r13
  __int64 v14; // r15
  __int64 *v15; // rax
  _QWORD *v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rbx
  __int64 v19; // [rsp+8h] [rbp-68h]
  __m128i v20; // [rsp+10h] [rbp-60h] BYREF
  __int64 v21; // [rsp+20h] [rbp-50h]
  __int64 v22; // [rsp+28h] [rbp-48h]
  __int64 v23; // [rsp+30h] [rbp-40h]

  v4 = *(_QWORD *)(a2 + 16);
  if ( *(_WORD *)(v4 + 68) && *(_WORD *)(v4 + 68) != 68 )
  {
    v10 = sub_3571960(a1, *(_QWORD *)(v4 + 24), 0);
  }
  else
  {
    v5 = *(_DWORD *)(v4 + 40) & 0xFFFFFF;
    if ( v5 == 1 )
LABEL_19:
      BUG();
    v6 = *(_QWORD *)(v4 + 32);
    if ( a2 == v6 + 40 )
    {
      v9 = 80;
    }
    else
    {
      v7 = 1;
      do
      {
        v8 = v7;
        v7 += 2;
        if ( v5 == v7 )
          goto LABEL_19;
      }
      while ( a2 != v6 + 40LL * v7 );
      v9 = 40LL * (v8 + 3);
    }
    v10 = sub_3571720(a1, *(_QWORD *)(v6 + v9 + 24), 0);
  }
  if ( v10 )
  {
    v11 = a1[1];
    if ( v11 )
    {
      if ( (v11 & 4) == 0 )
      {
        v12 = v11 & 0xFFFFFFFFFFFFFFF8LL;
        if ( v12 )
        {
          if ( !sub_2EBE590(a1[5], v10, v12, 0) )
          {
            v13 = *(_QWORD *)(v4 + 24);
            v14 = a1[5];
            v19 = a1[4];
            v15 = (__int64 *)sub_2E311E0(v13);
            v16 = sub_356E080(0x14u, v13, v15, a1[1], a1[2], v14, v19);
            v20.m128i_i64[0] = 0;
            v18 = v17;
            v20.m128i_i32[2] = v10;
            v21 = 0;
            v22 = 0;
            v23 = 0;
            sub_2E8EAD0(v17, (__int64)v16, &v20);
            v10 = *(_DWORD *)(*(_QWORD *)(v18 + 32) + 8LL);
          }
        }
      }
    }
  }
  sub_2EAB0C0(a2, v10);
}
