// Function: sub_31B9240
// Address: 0x31b9240
//
_BOOL8 __fastcall sub_31B9240(__int64 a1, __int64 a2, __int64 a3, int a4)
{
  _BOOL8 result; // rax
  unsigned __int16 v6; // ax
  char v7; // al
  unsigned __int8 *v8; // rsi
  _QWORD **v9; // rax
  __m128i v10; // xmm2
  __m128i v11; // xmm0
  __m128i v12; // xmm1
  int v13; // edx
  unsigned __int64 v14; // rax
  __int64 v15; // rcx
  __int64 v16; // rax
  int v17; // eax
  __m128i v18; // [rsp+0h] [rbp-A0h] BYREF
  __m128i v19; // [rsp+10h] [rbp-90h] BYREF
  __m128i v20; // [rsp+20h] [rbp-80h] BYREF
  char v21; // [rsp+30h] [rbp-70h]
  __m128i v22[3]; // [rsp+40h] [rbp-60h] BYREF
  char v23; // [rsp+70h] [rbp-30h]

  sub_D66840(&v18, *(_BYTE **)(a3 + 16));
  result = 1;
  if ( !v21 )
    return result;
  if ( sub_318B640(a2) )
  {
    if ( a2 )
      goto LABEL_4;
    sub_318B670(0);
LABEL_16:
    v8 = *(unsigned __int8 **)(a2 + 16);
    v13 = *v8;
    v14 = (unsigned int)(v13 - 29);
    if ( (unsigned int)v14 <= 0x38 )
    {
      v15 = 0x110000800000220LL;
      if ( _bittest64(&v15, v14) )
      {
        if ( (_BYTE)v13 != 85 )
          goto LABEL_6;
        v16 = *((_QWORD *)v8 - 4);
        if ( !v16 )
          goto LABEL_6;
        if ( *(_BYTE *)v16 )
          goto LABEL_6;
        if ( *(_QWORD *)(v16 + 24) != *((_QWORD *)v8 + 10) )
          goto LABEL_6;
        if ( (*(_BYTE *)(v16 + 33) & 0x20) == 0 )
          goto LABEL_6;
        v17 = *(_DWORD *)(v16 + 36);
        if ( v17 != 324 && v17 != 291 )
          goto LABEL_6;
      }
    }
LABEL_13:
    v9 = *(_QWORD ***)(a1 + 120);
    v10 = _mm_loadu_si128(&v20);
    v23 = 1;
    v11 = _mm_loadu_si128(&v18);
    v12 = _mm_loadu_si128(&v19);
    v22[2] = v10;
    v22[0] = v11;
    v22[1] = v12;
    v7 = sub_CF63E0(*v9, v8, v22, (__int64)(v9 + 1));
    goto LABEL_7;
  }
  if ( !sub_318B670(a2) || !a2 )
    goto LABEL_16;
LABEL_4:
  v6 = *(_WORD *)(*(_QWORD *)(a2 + 16) + 2LL);
  if ( ((v6 >> 7) & 6) == 0 && (v6 & 1) == 0 )
  {
    v8 = *(unsigned __int8 **)(a2 + 16);
    goto LABEL_13;
  }
LABEL_6:
  v7 = 3;
LABEL_7:
  if ( a4 <= 1 )
  {
    if ( a4 >= 0 )
      return (v7 & 2) != 0;
LABEL_27:
    BUG();
  }
  if ( a4 != 2 )
    goto LABEL_27;
  return v7 & 1;
}
