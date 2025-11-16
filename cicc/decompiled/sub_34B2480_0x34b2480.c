// Function: sub_34B2480
// Address: 0x34b2480
//
unsigned __int64 __fastcall sub_34B2480(unsigned __int64 *a1, int a2)
{
  bool v2; // r14
  __int64 v3; // rax
  __int64 v4; // r13
  int v5; // eax
  __int64 v6; // rax
  char v7; // cl
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax
  unsigned __int64 result; // rax
  __int64 v12; // rax
  char v13; // cl
  __int64 v14; // rax
  __int64 v15; // [rsp+0h] [rbp-30h] BYREF
  char v16; // [rsp+8h] [rbp-28h]

  if ( (unsigned __int16)(a2 - 17) > 0xD3u )
  {
    result = (unsigned int)(a2 - 1);
    if ( (unsigned __int16)(a2 - 1) > 0x110u
      || (_WORD)a2 == 270
      || (result = (unsigned int)(a2 - 229), (unsigned __int16)(a2 - 229) <= 0x1Fu) )
    {
      *a1 = 0;
      return result;
    }
    if ( (_WORD)a2 != 1 )
    {
      v12 = 16LL * ((unsigned __int16)a2 - 1);
      v13 = byte_444C4A0[v12 + 8];
      v14 = *(_QWORD *)&byte_444C4A0[v12];
      v16 = v13;
      v15 = v14;
      result = (sub_CA1930(&v15) << 32) | 1;
      *a1 = result;
      return result;
    }
LABEL_16:
    BUG();
  }
  v2 = 1;
  v3 = (unsigned __int16)a2 - 1;
  v4 = word_4456340[v3];
  if ( (unsigned __int16)v4 <= 1u )
    v2 = (unsigned __int16)(a2 - 176) <= 0x34u;
  v5 = (unsigned __int16)word_4456580[v3];
  if ( (unsigned __int16)v5 <= 1u || (unsigned __int16)(v5 - 504) <= 7u )
    goto LABEL_16;
  v6 = 16LL * (v5 - 1);
  v7 = byte_444C4A0[v6 + 8];
  v8 = *(_QWORD *)&byte_444C4A0[v6];
  v16 = v7;
  v15 = v8;
  v9 = sub_CA1930(&v15);
  *(_BYTE *)a1 = *(_BYTE *)a1 & 0xF8 | !v2 | (4 * v2);
  v10 = (v9 << 29) & 0x1FFFFFFFE0000000LL;
  if ( v2 )
    result = *a1 & 7 | (8 * (((unsigned __int16)(a2 - 176) <= 0x34u) | (unsigned __int64)(32 * v4) | v10));
  else
    result = *a1 & 7 | (8 * v10);
  *a1 = result;
  return result;
}
