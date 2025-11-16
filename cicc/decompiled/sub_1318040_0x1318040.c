// Function: sub_1318040
// Address: 0x1318040
//
__int64 __fastcall sub_1318040(
        __int64 a1,
        __int64 a2,
        unsigned __int64 a3,
        unsigned __int64 a4,
        unsigned __int8 a5,
        __int64 *a6)
{
  unsigned __int8 v6; // bl
  unsigned int v8; // r15d
  __int64 *v9; // r13
  void **v10; // rax
  void *v11; // r9
  void **v12; // r10
  unsigned int v14; // ecx
  unsigned __int64 v15; // r9
  __int64 v16; // rax
  unsigned __int64 v17; // [rsp+8h] [rbp-58h]
  __int64 v18; // [rsp+8h] [rbp-58h]
  unsigned __int8 v19; // [rsp+10h] [rbp-50h]
  __int64 *v20; // [rsp+10h] [rbp-50h]
  _BYTE v21[49]; // [rsp+2Fh] [rbp-31h] BYREF

  v6 = a5;
  if ( a3 > 0x3800 )
  {
    if ( a4 > 0x40 )
      return sub_1309830(a1, a2, a3, a4, a5);
    else
      return sub_1309DC0(a1, a2, a3, a5);
  }
  else
  {
    if ( a3 > 0x1000 )
    {
      _BitScanReverse64(&v15, 2 * a3 - 1);
      v8 = ((((a3 - 1) & (-1LL << ((unsigned __int8)v15 - 3))) >> ((unsigned __int8)v15 - 3)) & 3) + 4 * v15 - 23;
    }
    else
    {
      v8 = byte_5060800[(a3 + 7) >> 3];
    }
    if ( a6 )
    {
      v9 = &a6[3 * v8];
      v10 = (void **)v9[1];
      v11 = *v10;
      v12 = v10 + 1;
      if ( (_WORD)v10 != *((_WORD *)v9 + 12) )
      {
        v9[1] = (__int64)v12;
LABEL_7:
        if ( v6 )
          v11 = memset(v11, 0, qword_505FA40[v8]);
        ++v9[2];
        return (__int64)v11;
      }
      if ( (_WORD)v10 != *((_WORD *)v9 + 14) )
      {
        v9[1] = (__int64)v12;
        *((_WORD *)v9 + 12) = (_WORD)v12;
        goto LABEL_7;
      }
      v17 = a3;
      v19 = a5;
      v16 = sub_1314520(a1, a2);
      a5 = v19;
      a3 = v17;
      if ( !v16 )
        return 0;
      if ( *(_WORD *)(unk_5060A20 + 2LL * v8) )
      {
        v18 = v16;
        v20 = &a6[3 * v8 + 1];
        sub_1310140(a1, a6, v20, v8, 1);
        v11 = (void *)sub_13100A0(a1, v18, a6, v20, v8, v21);
        if ( v21[0] )
          goto LABEL_7;
        return 0;
      }
      v14 = v8;
      a2 = v16;
    }
    else
    {
      v14 = v8;
    }
    return sub_1317CF0(a1, a2, a3, v14, a5);
  }
}
