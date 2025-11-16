// Function: sub_34D43C0
// Address: 0x34d43c0
//
__int64 __fastcall sub_34D43C0(__int64 a1, int a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  __int64 v7; // r14
  __int64 v8; // rcx
  unsigned __int16 v9; // bx
  __int64 v10; // r13
  __int64 v11; // rdx
  __int64 v12; // r14
  unsigned __int16 v13; // si
  __int64 v14; // r13
  signed __int64 v15; // r15
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v19; // rax
  unsigned __int16 v20; // ax
  char v21; // al
  signed __int64 v22; // rax
  bool v23; // cc
  unsigned __int64 v24; // rax
  char v25; // [rsp+0h] [rbp-70h]
  unsigned __int64 v28; // [rsp+10h] [rbp-60h]
  signed __int64 v29; // [rsp+18h] [rbp-58h]
  __int64 v30; // [rsp+18h] [rbp-58h]
  __int64 v31; // [rsp+20h] [rbp-50h] BYREF
  __int64 v32; // [rsp+28h] [rbp-48h]
  __int64 v33; // [rsp+30h] [rbp-40h]

  if ( (unsigned __int16)sub_2D5BAE0(*(_QWORD *)(a1 + 32), *(_QWORD *)(a1 + 16), (__int64 *)a3, 1) == 1 )
    return 4;
  v7 = *(_QWORD *)a3;
  v29 = 1;
  v8 = sub_2D5BAE0(*(_QWORD *)(a1 + 32), *(_QWORD *)(a1 + 16), (__int64 *)a3, 0);
  v9 = v8;
  v10 = v7;
  v12 = v11;
  while ( 1 )
  {
    LOWORD(v8) = v9;
    sub_2FE6CC0((__int64)&v31, *(_QWORD *)(a1 + 32), v10, v8, v12);
    v13 = v32;
    if ( (_BYTE)v31 == 10 )
      break;
    if ( !(_BYTE)v31 )
    {
      v14 = a1;
      v15 = v29;
      v13 = v9;
      goto LABEL_9;
    }
    if ( (v31 & 0xFB) == 2 )
    {
      v19 = 2 * v29;
      if ( !is_mul_ok(2u, v29) )
      {
        v19 = 0x7FFFFFFFFFFFFFFFLL;
        if ( v29 <= 0 )
          v19 = 0x8000000000000000LL;
      }
      v29 = v19;
    }
    if ( v9 == (_WORD)v32 && ((_WORD)v32 || v33 == v12) )
    {
      v14 = a1;
      v15 = v29;
      goto LABEL_9;
    }
    v8 = v32;
    v12 = v33;
    v9 = v32;
  }
  v13 = 8;
  v14 = a1;
  if ( v9 )
    v13 = v9;
  v15 = 0;
LABEL_9:
  if ( !a6 && (unsigned int)*(unsigned __int8 *)(a3 + 8) - 17 <= 1 )
  {
    if ( v13 <= 1u || (unsigned __int16)(v13 - 504) <= 7u )
      BUG();
    v30 = *(_QWORD *)(v14 + 16);
    v28 = *(_QWORD *)&byte_444C4A0[16 * v13 - 16];
    v25 = byte_444C4A0[16 * v13 - 8];
    v16 = sub_9208B0(v30, a3);
    v32 = v17;
    v31 = v16;
    if ( (!(_BYTE)v17 || v25) && ((v16 + 7) & 0xFFFFFFFFFFFFFFF8LL) < v28 )
    {
      v20 = sub_2D5BAE0(*(_QWORD *)(v14 + 32), v30, (__int64 *)a3, 0);
      if ( a2 == 33 )
      {
        if ( v20 )
        {
          v21 = *(_BYTE *)(v20 + *(_QWORD *)(v14 + 32) + 274LL * v13 + 443718);
LABEL_25:
          if ( (v21 & 0xFB) == 0 )
            return v15;
        }
      }
      else if ( v20 )
      {
        v21 = (unsigned __int8)*(_WORD *)(*(_QWORD *)(v14 + 32) + 2 * (v20 + 274LL * v13 + 71704) + 6) >> 4;
        goto LABEL_25;
      }
      v22 = sub_34D2080(v14 + 8, a3, a2 != 33, a2 == 33);
      if ( __OFADD__(v22, v15) )
      {
        v23 = v22 <= 0;
        v24 = 0x8000000000000000LL;
        if ( !v23 )
          return 0x7FFFFFFFFFFFFFFFLL;
        return v24;
      }
      else
      {
        v15 += v22;
      }
    }
  }
  return v15;
}
