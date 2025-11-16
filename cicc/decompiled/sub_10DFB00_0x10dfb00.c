// Function: sub_10DFB00
// Address: 0x10dfb00
//
__int64 __fastcall sub_10DFB00(unsigned __int8 *a1, __m128i *a2)
{
  __int64 v3; // rax
  unsigned int v4; // edx
  unsigned int v5; // r12d
  __int64 v7; // rax
  unsigned __int64 v8; // rax
  __int64 v9; // rsi
  int v10; // eax
  __int64 v11; // rsi
  unsigned int v12; // eax
  __int64 v13; // [rsp+0h] [rbp-30h] BYREF
  unsigned int v14; // [rsp+8h] [rbp-28h]
  __int64 v15; // [rsp+10h] [rbp-20h]
  unsigned int v16; // [rsp+18h] [rbp-18h]

  sub_9AC330((__int64)&v13, (__int64)a1, 0, a2);
  v3 = 1LL << ((unsigned __int8)v14 - 1);
  if ( v14 > 0x40 )
  {
    v4 = v16;
    if ( (*(_QWORD *)(v13 + 8LL * ((v14 - 1) >> 6)) & v3) != 0 )
      goto LABEL_3;
LABEL_12:
    v7 = v15;
    if ( v4 > 0x40 )
      v7 = *(_QWORD *)(v15 + 8LL * ((v4 - 1) >> 6));
    if ( (v7 & (1LL << ((unsigned __int8)v4 - 1))) != 0 )
    {
      v5 = 257;
      goto LABEL_4;
    }
    v8 = *a1;
    if ( (unsigned __int8)v8 <= 0x1Cu )
    {
      if ( (_BYTE)v8 != 5 )
        goto LABEL_26;
      v10 = *((unsigned __int16 *)a1 + 1);
      if ( (*((_WORD *)a1 + 1) & 0xFFFD) != 0xD && (v10 & 0xFFF7) != 0x11 )
        goto LABEL_26;
    }
    else
    {
      if ( (unsigned __int8)v8 > 0x36u )
        goto LABEL_26;
      v9 = 0x40540000000000LL;
      if ( !_bittest64(&v9, v8) )
        goto LABEL_26;
      v10 = (unsigned __int8)v8 - 29;
    }
    if ( v10 == 15 && (a1[1] & 4) != 0 )
    {
      v11 = *((_QWORD *)a1 - 8);
      if ( v11 )
      {
        if ( *((_QWORD *)a1 - 4) )
        {
          LOWORD(v12) = sub_9A1D50(0x28u, v11, *((unsigned __int8 **)a1 - 4), a2[2].m128i_i64[1], a2->m128i_i64[0]);
          v4 = v16;
          v5 = v12;
          goto LABEL_4;
        }
      }
    }
LABEL_26:
    v5 = 0;
    goto LABEL_4;
  }
  v4 = v16;
  if ( (v13 & v3) == 0 )
    goto LABEL_12;
LABEL_3:
  v5 = 256;
LABEL_4:
  if ( v4 > 0x40 && v15 )
    j_j___libc_free_0_0(v15);
  if ( v14 > 0x40 && v13 )
    j_j___libc_free_0_0(v13);
  return v5;
}
