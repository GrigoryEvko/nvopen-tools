// Function: sub_BFA0F0
// Address: 0xbfa0f0
//
void __fastcall sub_BFA0F0(__int64 a1, __int64 a2)
{
  unsigned __int64 v3; // rcx
  __int64 v4; // r14
  int v5; // edx
  unsigned __int8 v6; // al
  const char *v7; // rax
  const char *v8; // rax
  __int64 v9; // r14
  _BYTE *v10; // rax
  __int64 v11; // rax
  __int64 v12; // rdi
  _QWORD v13[4]; // [rsp+0h] [rbp-50h] BYREF
  char v14; // [rsp+20h] [rbp-30h]
  char v15; // [rsp+21h] [rbp-2Fh]

  if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(a2 - 32) + 8LL) + 8LL) != 14 )
  {
    v15 = 1;
    v8 = "Store operand must be a pointer.";
    goto LABEL_19;
  }
  _BitScanReverse64(&v3, 1LL << (*(_WORD *)(a2 + 2) >> 1));
  if ( 0x8000000000000000LL >> ((unsigned __int8)v3 ^ 0x3Fu) > 0x100000000LL )
  {
    v15 = 1;
    v8 = "huge alignment values are unsupported";
LABEL_19:
    v9 = *(_QWORD *)a1;
    v13[0] = v8;
    v14 = 3;
    if ( !v9 )
    {
      *(_BYTE *)(a1 + 152) = 1;
      return;
    }
    sub_CA0E80(v13, v9);
    v10 = *(_BYTE **)(v9 + 32);
    if ( (unsigned __int64)v10 >= *(_QWORD *)(v9 + 24) )
    {
      sub_CB5D20(v9, 10);
    }
    else
    {
      *(_QWORD *)(v9 + 32) = v10 + 1;
      *v10 = 10;
    }
    v11 = *(_QWORD *)a1;
    *(_BYTE *)(a1 + 152) = 1;
    if ( v11 )
      goto LABEL_23;
    return;
  }
  v4 = *(_QWORD *)(*(_QWORD *)(a2 - 64) + 8LL);
  v5 = *(unsigned __int8 *)(v4 + 8);
  if ( (_BYTE)v5 != 12
    && (unsigned __int8)v5 > 3u
    && (_BYTE)v5 != 5
    && (v5 & 0xFD) != 4
    && (v5 & 0xFB) != 0xA
    && ((unsigned __int8)(*(_BYTE *)(v4 + 8) - 15) > 3u && v5 != 20 || !(unsigned __int8)sub_BCEBA0(v4, 0)) )
  {
    v15 = 1;
    v7 = "storing unsized types is not allowed";
    goto LABEL_16;
  }
  if ( !sub_B46500((unsigned __int8 *)a2) )
  {
    if ( *(_BYTE *)(a2 + 72) == 1 )
      goto LABEL_27;
    v15 = 1;
    v7 = "Non-atomic store cannot have SynchronizationScope specified";
LABEL_16:
    v13[0] = v7;
    v14 = 3;
    sub_BDBF70((__int64 *)a1, (__int64)v13);
    if ( !*(_QWORD *)a1 )
      return;
    goto LABEL_23;
  }
  if ( ((*(_WORD *)(a2 + 2) >> 7) & 5) == 4 )
  {
    v15 = 1;
    v7 = "Store cannot have Acquire ordering";
    goto LABEL_16;
  }
  v6 = *(_BYTE *)(v4 + 8);
  if ( (v6 & 0xFD) == 0xC || v6 <= 3u || v6 == 5 || (*(_BYTE *)(v4 + 8) & 0xFD) == 4 )
  {
    sub_BDBDF0(a1, v4, (_BYTE *)a2);
LABEL_27:
    sub_BF6FE0(a1, a2);
    return;
  }
  v15 = 1;
  v13[0] = "atomic store operand must have integer, pointer, or floating point type!";
  v14 = 3;
  sub_BDBF70((__int64 *)a1, (__int64)v13);
  v12 = *(_QWORD *)a1;
  if ( *(_QWORD *)a1 )
  {
    sub_BD9860(v12, v4);
LABEL_23:
    sub_BDBD80(a1, (_BYTE *)a2);
  }
}
