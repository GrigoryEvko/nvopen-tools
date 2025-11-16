// Function: sub_1758A30
// Address: 0x1758a30
//
_QWORD *__fastcall sub_1758A30(__int64 a1, __int64 a2, __int64 ***a3, __int64 a4)
{
  __int64 v5; // rsi
  _BYTE *v7; // rdi
  unsigned __int8 v8; // al
  _BYTE *v9; // r12
  unsigned int v10; // eax
  __int64 v11; // rsi
  __int64 v12; // rcx
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // r13
  __int64 v16; // rax
  __int64 v17; // rbx
  _QWORD *v18; // r12
  _QWORD **v19; // rax
  __int16 v20; // r14
  _QWORD *v21; // r15
  __int64 *v22; // rax
  __int64 v23; // rsi
  __int64 v25; // rax
  unsigned int v27; // [rsp+1Ch] [rbp-54h] BYREF
  _BYTE v28[16]; // [rsp+20h] [rbp-50h] BYREF
  __int16 v29; // [rsp+30h] [rbp-40h]

  v5 = a4;
  v7 = *(a3 - 3);
  v8 = v7[16];
  v9 = v7 + 24;
  if ( v8 != 13 )
  {
    if ( *(_BYTE *)(*(_QWORD *)v7 + 8LL) != 16 )
      return 0;
    if ( v8 > 0x10u )
      return 0;
    v25 = sub_15A1020(v7, a4, *(_QWORD *)v7, a4);
    if ( !v25 || *(_BYTE *)(v25 + 16) != 13 )
      return 0;
    v5 = a4;
    v9 = (_BYTE *)(v25 + 24);
  }
  v10 = *(unsigned __int16 *)(a2 + 18);
  BYTE1(v10) &= ~0x80u;
  v27 = v10;
  if ( !(unsigned __int8)sub_1757250((int *)&v27, v5) || !sub_15F2380((__int64)a3) )
    return 0;
  v11 = *((unsigned int *)v9 + 2);
  v12 = (unsigned int)(v11 - 1);
  v13 = 1LL << ((unsigned __int8)v11 - 1);
  v14 = *(_QWORD *)v9;
  if ( (unsigned int)v11 > 0x40 )
  {
    v12 = (unsigned int)v12 >> 6;
    if ( (*(_QWORD *)(v14 + 8 * v12) & v13) == 0 )
      goto LABEL_6;
  }
  else if ( (v14 & v13) == 0 )
  {
    goto LABEL_6;
  }
  v27 = sub_15FF5D0(v27);
LABEL_6:
  v15 = (__int64)*(a3 - 6);
  v16 = sub_15A06D0(*a3, v11, v13, v12);
  v29 = 257;
  v17 = v16;
  v18 = sub_1648A60(56, 2u);
  if ( v18 )
  {
    v19 = *(_QWORD ***)v15;
    v20 = v27;
    if ( *(_BYTE *)(*(_QWORD *)v15 + 8LL) == 16 )
    {
      v21 = v19[4];
      v22 = (__int64 *)sub_1643320(*v19);
      v23 = (__int64)sub_16463B0(v22, (unsigned int)v21);
    }
    else
    {
      v23 = sub_1643320(*v19);
    }
    sub_15FEC10((__int64)v18, v23, 51, v20, v15, v17, (__int64)v28, 0);
  }
  return v18;
}
