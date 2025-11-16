// Function: sub_30670E0
// Address: 0x30670e0
//
__int64 __fastcall sub_30670E0(__int64 a1, int a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  __int64 v7; // r14
  __int64 v8; // rcx
  unsigned __int16 v9; // bx
  __int64 v10; // r12
  __int64 v11; // rdx
  __int64 v12; // r14
  unsigned __int16 v13; // dx
  __int64 v14; // r12
  unsigned __int64 v15; // rbx
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v19; // rax
  unsigned __int16 v20; // ax
  char v21; // al
  unsigned int v22; // eax
  unsigned __int64 v23; // rdx
  bool v24; // zf
  unsigned __int64 v25; // rax
  signed __int64 v26; // r12
  signed __int64 v27; // rbx
  unsigned __int64 v28; // rax
  char v29; // [rsp+7h] [rbp-69h]
  __int64 v30; // [rsp+8h] [rbp-68h]
  unsigned int v32; // [rsp+14h] [rbp-5Ch]
  signed __int64 v33; // [rsp+18h] [rbp-58h]
  unsigned __int64 v34; // [rsp+20h] [rbp-50h] BYREF
  __int64 v35; // [rsp+28h] [rbp-48h]
  __int64 v36; // [rsp+30h] [rbp-40h]

  if ( (unsigned __int16)sub_2D5BAE0(*(_QWORD *)(a1 + 24), *(_QWORD *)(a1 + 8), (__int64 *)a3, 1) != 1 )
  {
    v7 = *(_QWORD *)a3;
    v33 = 1;
    v8 = sub_2D5BAE0(*(_QWORD *)(a1 + 24), *(_QWORD *)(a1 + 8), (__int64 *)a3, 0);
    v9 = v8;
    v10 = v7;
    v12 = v11;
    while ( 1 )
    {
      LOWORD(v8) = v9;
      sub_2FE6CC0((__int64)&v34, *(_QWORD *)(a1 + 24), v10, v8, v12);
      v13 = v35;
      if ( (_BYTE)v34 == 10 )
        break;
      if ( !(_BYTE)v34 )
      {
        v14 = a1;
        v13 = v9;
        goto LABEL_9;
      }
      if ( (v34 & 0xFB) == 2 )
      {
        v19 = 2 * v33;
        if ( !is_mul_ok(2u, v33) )
        {
          v19 = 0x7FFFFFFFFFFFFFFFLL;
          if ( v33 <= 0 )
            v19 = 0x8000000000000000LL;
        }
        v33 = v19;
      }
      if ( v9 == (_WORD)v35 && ((_WORD)v35 || v12 == v36) )
      {
        v14 = a1;
        goto LABEL_9;
      }
      v8 = v35;
      v12 = v36;
      v9 = v35;
    }
    v13 = 8;
    v14 = a1;
    v33 = 0;
    if ( v9 )
      v13 = v9;
LABEL_9:
    if ( a6 || (unsigned int)*(unsigned __int8 *)(a3 + 8) - 17 > 1 )
      return v33;
    v32 = v13;
    if ( v13 <= 1u || (unsigned __int16)(v13 - 504) <= 7u )
      BUG();
    v30 = *(_QWORD *)(v14 + 8);
    v15 = *(_QWORD *)&byte_444C4A0[16 * v13 - 16];
    v29 = byte_444C4A0[16 * v13 - 8];
    v16 = sub_9208B0(v30, a3);
    v35 = v17;
    v34 = v16;
    if ( (_BYTE)v17 )
    {
      if ( !v29 )
        return v33;
    }
    if ( ((v16 + 7) & 0xFFFFFFFFFFFFFFF8LL) >= v15 )
      return v33;
    v20 = sub_2D5BAE0(*(_QWORD *)(v14 + 24), v30, (__int64 *)a3, 0);
    if ( a2 == 33 )
    {
      if ( v20 )
      {
        v21 = *(_BYTE *)(v20 + *(_QWORD *)(v14 + 24) + 274LL * v32 + 443718);
LABEL_25:
        if ( (v21 & 0xFB) == 0 )
          return v33;
      }
    }
    else if ( v20 )
    {
      v21 = (unsigned __int8)*(_WORD *)(*(_QWORD *)(v14 + 24) + 2 * (v20 + 274LL * v32 + 71704) + 6) >> 4;
      goto LABEL_25;
    }
    if ( *(_BYTE *)(a3 + 8) == 18 )
      return v33;
    v22 = *(_DWORD *)(a3 + 32);
    LODWORD(v35) = v22;
    if ( v22 > 0x40 )
    {
      sub_C43690((__int64)&v34, -1, 1);
    }
    else
    {
      v23 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v22;
      v24 = v22 == 0;
      v25 = 0;
      if ( !v24 )
        v25 = v23;
      v34 = v25;
    }
    v26 = sub_3064F80(v14, a3, (__int64 *)&v34, a2 != 33, a2 == 33);
    if ( (unsigned int)v35 > 0x40 && v34 )
      j_j___libc_free_0_0(v34);
    v27 = v26 + v33;
    if ( __OFADD__(v26, v33) )
    {
      v28 = 0x8000000000000000LL;
      if ( v26 > 0 )
        return 0x7FFFFFFFFFFFFFFFLL;
      return v28;
    }
    return v27;
  }
  return 4;
}
