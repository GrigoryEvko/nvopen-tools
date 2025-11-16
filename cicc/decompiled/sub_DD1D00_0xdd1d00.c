// Function: sub_DD1D00
// Address: 0xdd1d00
//
_QWORD *__fastcall sub_DD1D00(__int64 *a1, _BYTE *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r13
  __int16 v6; // ax
  __int64 v7; // rax
  _QWORD *result; // rax
  unsigned __int16 v9; // ax
  __int64 *v10; // rbx
  __int64 v11; // rdx
  __int64 v12; // r9
  __int64 v13; // r12
  __int64 v14; // r12
  __int64 v15; // rax
  unsigned __int64 v16; // rdx
  __int64 v17; // r12
  __int64 v18; // rax
  __int64 v19; // rax
  _QWORD *v20; // rax
  __int64 v21; // rsi
  __int64 v22; // [rsp+8h] [rbp-58h]
  _QWORD *v23; // [rsp+8h] [rbp-58h]
  _BYTE *v24; // [rsp+10h] [rbp-50h] BYREF
  __int64 v25; // [rsp+18h] [rbp-48h]
  _BYTE v26[64]; // [rsp+20h] [rbp-40h] BYREF

  v5 = (__int64)a2;
  v6 = *((_WORD *)a2 + 12);
  if ( v6 )
  {
    v9 = v6 - 9;
    if ( v9 > 3u )
      goto LABEL_18;
    v10 = (__int64 *)*((_QWORD *)a2 + 4);
    v11 = *((_QWORD *)a2 + 5);
    v24 = v26;
    v12 = (__int64)&v10[v11];
    v25 = 0x200000000LL;
    if ( v10 != (__int64 *)v12 )
    {
      do
      {
        v17 = *v10;
        v22 = v12;
        if ( *(_WORD *)(*v10 + 24) != 5
          || *(_QWORD *)(v17 + 40) != 2
          || !sub_D96960(**(_QWORD **)(v17 + 32))
          || (v13 = *(_QWORD *)(*(_QWORD *)(v17 + 32) + 8LL), *(_WORD *)(v13 + 24) != 6)
          || *(_QWORD *)(v13 + 40) != 2
          || !sub_D96960(**(_QWORD **)(v13 + 32))
          || (v14 = *(_QWORD *)(*(_QWORD *)(v13 + 32) + 8LL)) == 0 )
        {
          if ( v24 != v26 )
            _libc_free(v24, a2);
          goto LABEL_18;
        }
        v15 = (unsigned int)v25;
        v12 = v22;
        v16 = (unsigned int)v25 + 1LL;
        if ( v16 > HIDWORD(v25) )
        {
          a2 = v26;
          sub_C8D5F0((__int64)&v24, v26, v16, 8u, a5, v22);
          v15 = (unsigned int)v25;
          v12 = v22;
        }
        ++v10;
        *(_QWORD *)&v24[8 * v15] = v14;
        LODWORD(v25) = v25 + 1;
      }
      while ( (__int64 *)v12 != v10 );
      v9 = *(_WORD *)(v5 + 24) - 9;
      if ( v9 > 3u )
        BUG();
    }
    v21 = word_3F74E78[v9];
    result = (_QWORD *)sub_DCD310(a1, v21, (__int64)&v24, (__int64)word_3F74E78, a5);
    if ( v24 != v26 )
    {
      v23 = result;
      _libc_free(v24, v21);
      result = v23;
    }
    if ( !result )
    {
LABEL_18:
      v18 = sub_D95540(v5);
      v19 = sub_D97090((__int64)a1, v18);
      v20 = sub_DA2C50((__int64)a1, v19, -1, 1u);
      return sub_DCC810(a1, (__int64)v20, v5, 0, 0);
    }
  }
  else
  {
    v7 = sub_AD63D0(*((_QWORD *)a2 + 4));
    return sub_DA2570((__int64)a1, v7);
  }
  return result;
}
