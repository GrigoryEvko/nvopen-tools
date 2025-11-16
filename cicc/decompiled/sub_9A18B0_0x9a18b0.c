// Function: sub_9A18B0
// Address: 0x9a18b0
//
__int16 __fastcall sub_9A18B0(__int64 a1, _BYTE *a2, __int64 a3, unsigned __int8 a4, int a5)
{
  _BYTE *v7; // rbx
  char v9; // r15
  char v10; // al
  unsigned __int16 v11; // dx
  __int16 result; // ax
  char v13; // al
  __int64 v14; // rsi
  int *v15; // rax
  int v16; // eax
  _BYTE *v17; // rbx
  __int64 v18; // rdi
  _BYTE *v19; // rdi
  __int64 v20; // rbx
  __int64 v21; // rsi
  unsigned __int16 v22; // ax
  __int64 v23; // rax
  unsigned __int16 v24; // ax
  _BYTE *v25; // rbx
  __int64 v26; // rdi
  _BYTE *v27; // rdi
  __int64 v28; // rbx
  __int64 v29; // rsi
  unsigned __int16 v30; // ax
  __int64 *v31; // rbx
  __int64 *v32; // rbx
  unsigned __int16 v33; // ax
  __int64 v34; // [rsp+8h] [rbp-58h]
  __int64 v35; // [rsp+8h] [rbp-58h]
  _BYTE *v37; // [rsp+18h] [rbp-48h] BYREF
  _QWORD *v38; // [rsp+20h] [rbp-40h] BYREF
  __int64 *v39; // [rsp+28h] [rbp-38h]

  v37 = a2;
  if ( a2 == (_BYTE *)a1 )
    goto LABEL_16;
  v7 = a2;
  v38 = 0;
  v39 = (__int64 *)&v37;
  if ( *a2 == 59 )
  {
    v13 = sub_995B10(&v38, *((_QWORD *)a2 - 8));
    v14 = *((_QWORD *)a2 - 4);
    if ( v13 && v14 )
    {
      *v39 = v14;
    }
    else
    {
      v9 = sub_995B10(&v38, v14);
      if ( !v9 )
      {
        v7 = v37;
        goto LABEL_4;
      }
      v23 = *((_QWORD *)v7 - 8);
      if ( !v23 )
      {
        v7 = v37;
        goto LABEL_3;
      }
      *v39 = v23;
    }
    v7 = v37;
    v9 = 1;
    if ( v37 != (_BYTE *)a1 )
      goto LABEL_4;
    a4 ^= 1u;
LABEL_16:
    LOBYTE(result) = a4;
    HIBYTE(result) = 1;
    return result;
  }
LABEL_3:
  v9 = 0;
LABEL_4:
  v10 = *v7;
  if ( *v7 <= 0x1Cu )
    goto LABEL_18;
  if ( v10 == 83 )
    return sub_9A13D0(a1, *((_WORD *)v7 + 1) & 0x3F, *((_QWORD *)v7 - 8), *((unsigned __int8 **)v7 - 4), a3, a4, a5);
  if ( v10 != 82 )
  {
LABEL_18:
    v15 = (int *)sub_C94E20(qword_4F862D0);
    if ( v15 )
      v16 = *v15;
    else
      v16 = qword_4F862D0[2];
    if ( a5 == v16 )
      return 0;
    v17 = v37;
    if ( *v37 <= 0x1Cu )
      return 0;
    v18 = *((_QWORD *)v37 + 1);
    if ( (unsigned int)*(unsigned __int8 *)(v18 + 8) - 17 <= 1 )
      v18 = **(_QWORD **)(v18 + 16);
    if ( (unsigned __int8)sub_BCAC40(v18, 1) )
    {
      if ( *v17 == 58 )
      {
        if ( (v17[7] & 0x40) != 0 )
          v31 = (__int64 *)*((_QWORD *)v17 - 1);
        else
          v31 = (__int64 *)&v17[-32 * (*((_DWORD *)v17 + 1) & 0x7FFFFFF)];
        v21 = *v31;
        if ( !*v31 )
          goto LABEL_42;
        v20 = v31[4];
        if ( !v20 )
          goto LABEL_42;
      }
      else
      {
        if ( *v17 != 86 )
          goto LABEL_42;
        v34 = *((_QWORD *)v17 - 12);
        if ( *(_QWORD *)(v34 + 8) != *((_QWORD *)v17 + 1) )
          goto LABEL_42;
        v19 = (_BYTE *)*((_QWORD *)v17 - 8);
        if ( *v19 > 0x15u )
          goto LABEL_42;
        v20 = *((_QWORD *)v17 - 4);
        if ( !(unsigned __int8)sub_AD7A80(v19) )
          goto LABEL_42;
        v21 = v34;
        if ( !v20 )
          goto LABEL_42;
      }
      v22 = sub_9A18B0(a1, v21, a3);
      if ( HIBYTE(v22) && (_BYTE)v22 || (v24 = sub_9A18B0(a1, v20, a3), HIBYTE(v24)) && (_BYTE)v24 )
      {
        LOBYTE(result) = v9 ^ 1;
        HIBYTE(result) = 1;
        return result;
      }
    }
LABEL_42:
    v25 = v37;
    if ( *v37 <= 0x1Cu )
      return 0;
    v26 = *((_QWORD *)v37 + 1);
    if ( (unsigned int)*(unsigned __int8 *)(v26 + 8) - 17 <= 1 )
      v26 = **(_QWORD **)(v26 + 16);
    if ( !(unsigned __int8)sub_BCAC40(v26, 1) )
      return 0;
    if ( *v25 == 57 )
    {
      if ( (v25[7] & 0x40) != 0 )
        v32 = (__int64 *)*((_QWORD *)v25 - 1);
      else
        v32 = (__int64 *)&v25[-32 * (*((_DWORD *)v25 + 1) & 0x7FFFFFF)];
      v29 = *v32;
      if ( !*v32 )
        return 0;
      v28 = v32[4];
      if ( !v28 )
        return 0;
    }
    else
    {
      if ( *v25 != 86 )
        return 0;
      v35 = *((_QWORD *)v25 - 12);
      if ( *(_QWORD *)(v35 + 8) != *((_QWORD *)v25 + 1) )
        return 0;
      v27 = (_BYTE *)*((_QWORD *)v25 - 4);
      if ( *v27 > 0x15u )
        return 0;
      v28 = *((_QWORD *)v25 - 8);
      if ( !(unsigned __int8)sub_AC30F0(v27) )
        return 0;
      v29 = v35;
      if ( !v28 )
        return 0;
    }
    v30 = sub_9A18B0(a1, v29, a3);
    if ( HIBYTE(v30) && !(_BYTE)v30 || (v33 = sub_9A18B0(a1, v28, a3), HIBYTE(v33)) && !(_BYTE)v33 )
    {
      LOBYTE(result) = v9;
      HIBYTE(result) = 1;
      return result;
    }
    return 0;
  }
  v11 = sub_9A13D0(
          a1,
          *((_WORD *)v7 + 1) & 0x3F | ((unsigned __int64)((v7[1] & 2) != 0) << 32),
          *((_QWORD *)v7 - 8),
          *((unsigned __int8 **)v7 - 4),
          a3,
          a4,
          a5);
  result = 0;
  if ( HIBYTE(v11) )
  {
    if ( v9 )
      LOBYTE(v11) = v11 ^ 1;
    LOBYTE(result) = v11;
    HIBYTE(result) = 1;
  }
  return result;
}
