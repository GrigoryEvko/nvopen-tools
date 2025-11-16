// Function: sub_3065270
// Address: 0x3065270
//
__int64 __fastcall sub_3065270(__int64 a1, int a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  __int64 v7; // r14
  __int64 v8; // rcx
  __int64 v9; // rdx
  __int64 v10; // rbx
  unsigned __int16 i; // r13
  unsigned __int16 v12; // dx
  __int64 v13; // rbx
  signed __int64 v14; // r15
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v18; // rax
  unsigned __int16 v19; // ax
  char v20; // al
  unsigned int v21; // eax
  unsigned __int64 v22; // rdx
  bool v23; // zf
  unsigned __int64 v24; // rax
  signed __int64 v25; // rbx
  signed __int64 v26; // r10
  unsigned __int64 v27; // rax
  unsigned __int64 v28; // [rsp+0h] [rbp-70h]
  char v29; // [rsp+Bh] [rbp-65h]
  __int64 v31; // [rsp+10h] [rbp-60h]
  signed __int64 v32; // [rsp+18h] [rbp-58h]
  unsigned int v33; // [rsp+18h] [rbp-58h]
  unsigned __int64 v34; // [rsp+20h] [rbp-50h] BYREF
  __int64 v35; // [rsp+28h] [rbp-48h]
  __int64 v36; // [rsp+30h] [rbp-40h]

  if ( (unsigned __int16)sub_2D5BAE0(*(_QWORD *)(a1 + 32), *(_QWORD *)(a1 + 16), (__int64 *)a3, 1) == 1 )
    return 4;
  v32 = 1;
  v7 = *(_QWORD *)a3;
  v8 = sub_2D5BAE0(*(_QWORD *)(a1 + 32), *(_QWORD *)(a1 + 16), (__int64 *)a3, 0);
  v10 = v9;
  for ( i = v8; ; i = v35 )
  {
    LOWORD(v8) = i;
    sub_2FE6CC0((__int64)&v34, *(_QWORD *)(a1 + 32), v7, v8, v10);
    v12 = v35;
    if ( (_BYTE)v34 == 10 )
      break;
    if ( !(_BYTE)v34 )
    {
      v13 = a1;
      v12 = i;
      v14 = v32;
      goto LABEL_9;
    }
    if ( (v34 & 0xFB) == 2 )
    {
      v18 = 2 * v32;
      if ( !is_mul_ok(2u, v32) )
      {
        v18 = 0x7FFFFFFFFFFFFFFFLL;
        if ( v32 <= 0 )
          v18 = 0x8000000000000000LL;
      }
      v32 = v18;
    }
    if ( i == (_WORD)v35 && ((_WORD)v35 || v10 == v36) )
    {
      v13 = a1;
      v14 = v32;
      goto LABEL_9;
    }
    v8 = v35;
    v10 = v36;
  }
  v12 = 8;
  v13 = a1;
  if ( i )
    v12 = i;
  v14 = 0;
LABEL_9:
  if ( !a6 && (unsigned int)*(unsigned __int8 *)(a3 + 8) - 17 <= 1 )
  {
    v33 = v12;
    if ( v12 <= 1u || (unsigned __int16)(v12 - 504) <= 7u )
      BUG();
    v31 = *(_QWORD *)(v13 + 16);
    v28 = *(_QWORD *)&byte_444C4A0[16 * v12 - 16];
    v29 = byte_444C4A0[16 * v12 - 8];
    v15 = sub_9208B0(v31, a3);
    v35 = v16;
    v34 = v15;
    if ( (!(_BYTE)v16 || v29) && ((v15 + 7) & 0xFFFFFFFFFFFFFFF8LL) < v28 )
    {
      v19 = sub_2D5BAE0(*(_QWORD *)(v13 + 32), v31, (__int64 *)a3, 0);
      if ( a2 == 33 )
      {
        if ( !v19 )
          goto LABEL_26;
        v20 = *(_BYTE *)(v19 + *(_QWORD *)(v13 + 32) + 274LL * v33 + 443718);
      }
      else
      {
        if ( !v19 )
          goto LABEL_26;
        v20 = (unsigned __int8)*(_WORD *)(*(_QWORD *)(v13 + 32) + 2 * (v19 + 274LL * v33 + 71704) + 6) >> 4;
      }
      if ( (v20 & 0xFB) == 0 )
        return v14;
LABEL_26:
      if ( *(_BYTE *)(a3 + 8) == 18 )
        return v14;
      v21 = *(_DWORD *)(a3 + 32);
      LODWORD(v35) = v21;
      if ( v21 > 0x40 )
      {
        sub_C43690((__int64)&v34, -1, 1);
      }
      else
      {
        v22 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v21;
        v23 = v21 == 0;
        v24 = 0;
        if ( !v23 )
          v24 = v22;
        v34 = v24;
      }
      v25 = sub_3064F80(v13 + 8, a3, (__int64 *)&v34, a2 != 33, a2 == 33);
      if ( (unsigned int)v35 > 0x40 && v34 )
        j_j___libc_free_0_0(v34);
      v26 = v25 + v14;
      if ( __OFADD__(v25, v14) )
      {
        v27 = 0x8000000000000000LL;
        if ( v25 > 0 )
          return 0x7FFFFFFFFFFFFFFFLL;
        return v27;
      }
      return v26;
    }
  }
  return v14;
}
