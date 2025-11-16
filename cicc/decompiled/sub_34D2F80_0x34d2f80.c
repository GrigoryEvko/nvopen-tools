// Function: sub_34D2F80
// Address: 0x34d2f80
//
__int64 __fastcall sub_34D2F80(__int64 a1, int a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  __int64 v7; // r14
  __int64 v8; // rcx
  unsigned __int16 v9; // bx
  __int64 v10; // rdx
  __int64 v11; // r13
  unsigned int v12; // r13d
  __int64 v13; // r14
  char v14; // bl
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v18; // rax
  unsigned __int16 v19; // ax
  char v20; // al
  unsigned __int64 v21; // rax
  __int64 v22; // rcx
  unsigned __int64 v23; // rax
  unsigned __int64 v24; // [rsp+8h] [rbp-68h]
  signed __int64 v26; // [rsp+18h] [rbp-58h]
  __int64 v27; // [rsp+20h] [rbp-50h] BYREF
  __int64 v28; // [rsp+28h] [rbp-48h]
  __int64 v29; // [rsp+30h] [rbp-40h]

  if ( (unsigned __int16)sub_2D5BAE0(*(_QWORD *)(a1 + 24), *(_QWORD *)(a1 + 8), (__int64 *)a3, 1) != 1 )
  {
    v7 = *(_QWORD *)a3;
    v26 = 1;
    v8 = sub_2D5BAE0(*(_QWORD *)(a1 + 24), *(_QWORD *)(a1 + 8), (__int64 *)a3, 0);
    v9 = v8;
    v11 = v10;
    while ( 1 )
    {
      LOWORD(v8) = v9;
      sub_2FE6CC0((__int64)&v27, *(_QWORD *)(a1 + 24), v7, v8, v11);
      if ( (_BYTE)v27 == 10 )
        break;
      if ( !(_BYTE)v27 )
        goto LABEL_8;
      if ( (v27 & 0xFB) == 2 )
      {
        v18 = 2 * v26;
        if ( !is_mul_ok(2u, v26) )
        {
          v18 = 0x7FFFFFFFFFFFFFFFLL;
          if ( v26 <= 0 )
            v18 = 0x8000000000000000LL;
        }
        v26 = v18;
      }
      if ( (_WORD)v28 == v9 && ((_WORD)v28 || v11 == v29) )
        goto LABEL_8;
      v8 = v28;
      v11 = v29;
      v9 = v28;
    }
    v26 = 0;
    if ( !v9 )
      v9 = 8;
LABEL_8:
    if ( a6 || (unsigned int)*(unsigned __int8 *)(a3 + 8) - 17 > 1 )
      return v26;
    v12 = v9;
    if ( v9 <= 1u || (unsigned __int16)(v9 - 504) <= 7u )
      BUG();
    v13 = *(_QWORD *)(a1 + 8);
    v24 = *(_QWORD *)&byte_444C4A0[16 * v9 - 16];
    v14 = byte_444C4A0[16 * v9 - 8];
    v15 = sub_9208B0(v13, a3);
    v28 = v16;
    v27 = v15;
    if ( (_BYTE)v16 )
    {
      if ( !v14 )
        return v26;
    }
    if ( ((v15 + 7) & 0xFFFFFFFFFFFFFFF8LL) >= v24 )
      return v26;
    v19 = sub_2D5BAE0(*(_QWORD *)(a1 + 24), v13, (__int64 *)a3, 0);
    if ( a2 == 33 )
    {
      if ( !v19 )
        goto LABEL_25;
      v20 = *(_BYTE *)(v19 + *(_QWORD *)(a1 + 24) + 274LL * v12 + 443718);
    }
    else
    {
      if ( !v19 )
        goto LABEL_25;
      v20 = (unsigned __int8)*(_WORD *)(*(_QWORD *)(a1 + 24) + 2 * (v19 + 274LL * v12 + 71704) + 6) >> 4;
    }
    if ( (v20 & 0xFB) == 0 )
      return v26;
LABEL_25:
    v21 = sub_34D2080(a1, a3, a2 != 33, a2 == 33);
    v22 = v21;
    if ( __OFADD__(v21, v26) )
    {
      v23 = 0x8000000000000000LL;
      if ( v22 > 0 )
        return 0x7FFFFFFFFFFFFFFFLL;
      return v23;
    }
    else
    {
      v26 += v21;
    }
    return v26;
  }
  return 4;
}
