// Function: sub_3072C00
// Address: 0x3072c00
//
__int64 __fastcall sub_3072C00(__int64 a1, int a2, __int64 a3, int a4)
{
  __int64 v5; // r14
  __int64 v6; // rcx
  unsigned __int16 v7; // bx
  __int64 v8; // rdx
  __int64 v9; // r13
  unsigned int v10; // r13d
  __int64 v11; // r14
  char v12; // bl
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v16; // rax
  unsigned __int16 v17; // ax
  char v18; // al
  unsigned __int64 v19; // rax
  __int64 v20; // rcx
  unsigned __int64 v21; // rax
  unsigned __int64 v22; // [rsp+8h] [rbp-68h]
  signed __int64 v24; // [rsp+18h] [rbp-58h]
  __int64 v25; // [rsp+20h] [rbp-50h] BYREF
  __int64 v26; // [rsp+28h] [rbp-48h]
  __int64 v27; // [rsp+30h] [rbp-40h]

  if ( (unsigned __int16)sub_2D5BAE0(*(_QWORD *)(a1 + 24), *(_QWORD *)(a1 + 8), (__int64 *)a3, 1) != 1 )
  {
    v5 = *(_QWORD *)a3;
    v24 = 1;
    v6 = sub_2D5BAE0(*(_QWORD *)(a1 + 24), *(_QWORD *)(a1 + 8), (__int64 *)a3, 0);
    v7 = v6;
    v9 = v8;
    while ( 1 )
    {
      LOWORD(v6) = v7;
      sub_2FE6CC0((__int64)&v25, *(_QWORD *)(a1 + 24), v5, v6, v9);
      if ( (_BYTE)v25 == 10 )
        break;
      if ( !(_BYTE)v25 )
        goto LABEL_8;
      if ( (v25 & 0xFB) == 2 )
      {
        v16 = 2 * v24;
        if ( !is_mul_ok(2u, v24) )
        {
          v16 = 0x7FFFFFFFFFFFFFFFLL;
          if ( v24 <= 0 )
            v16 = 0x8000000000000000LL;
        }
        v24 = v16;
      }
      if ( (_WORD)v26 == v7 && ((_WORD)v26 || v9 == v27) )
        goto LABEL_8;
      v6 = v26;
      v9 = v27;
      v7 = v26;
    }
    v24 = 0;
    if ( !v7 )
      v7 = 8;
LABEL_8:
    if ( a4 || (unsigned int)*(unsigned __int8 *)(a3 + 8) - 17 > 1 )
      return v24;
    v10 = v7;
    if ( v7 <= 1u || (unsigned __int16)(v7 - 504) <= 7u )
      BUG();
    v11 = *(_QWORD *)(a1 + 8);
    v22 = *(_QWORD *)&byte_444C4A0[16 * v7 - 16];
    v12 = byte_444C4A0[16 * v7 - 8];
    v13 = sub_9208B0(v11, a3);
    v26 = v14;
    v25 = v13;
    if ( (_BYTE)v14 )
    {
      if ( !v12 )
        return v24;
    }
    if ( ((v13 + 7) & 0xFFFFFFFFFFFFFFF8LL) >= v22 )
      return v24;
    v17 = sub_2D5BAE0(*(_QWORD *)(a1 + 24), v11, (__int64 *)a3, 0);
    if ( a2 == 33 )
    {
      if ( !v17 )
        goto LABEL_25;
      v18 = *(_BYTE *)(v17 + *(_QWORD *)(a1 + 24) + 274LL * v10 + 443718);
    }
    else
    {
      if ( !v17 )
        goto LABEL_25;
      v18 = (unsigned __int8)*(_WORD *)(*(_QWORD *)(a1 + 24) + 2 * (v17 + 274LL * v10 + 71704) + 6) >> 4;
    }
    if ( (v18 & 0xFB) == 0 )
      return v24;
LABEL_25:
    v19 = sub_30727B0(a1, a3, a2 != 33, a2 == 33);
    v20 = v19;
    if ( __OFADD__(v19, v24) )
    {
      v21 = 0x8000000000000000LL;
      if ( v20 > 0 )
        return 0x7FFFFFFFFFFFFFFFLL;
      return v21;
    }
    else
    {
      v24 += v19;
    }
    return v24;
  }
  return 4;
}
