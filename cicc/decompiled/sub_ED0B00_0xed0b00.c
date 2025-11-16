// Function: sub_ED0B00
// Address: 0xed0b00
//
__int64 *__fastcall sub_ED0B00(__int64 *a1, __int64 a2, char a3, __int64 a4)
{
  __int64 v4; // rax
  char v5; // bl
  __int64 v6; // r14
  unsigned __int64 v7; // rdx
  unsigned __int64 v8; // r15
  char *v9; // rax
  unsigned __int64 v10; // rdx
  _BYTE *v12; // rsi
  __int64 v13; // rdx
  char *v14; // rax
  unsigned __int64 v15; // rdx
  _QWORD v16[2]; // [rsp+0h] [rbp-60h] BYREF
  __int64 v17; // [rsp+10h] [rbp-50h] BYREF
  char v18; // [rsp+20h] [rbp-40h]

  if ( a3 )
  {
    sub_ED0A20((__int64)v16, a4);
    if ( v18 )
    {
      v12 = (_BYTE *)v16[0];
      v13 = v16[1];
      *a1 = (__int64)(a1 + 2);
      sub_ED0570(a1, v12, (__int64)&v12[v13]);
      if ( v18 )
      {
        v18 = 0;
        if ( (__int64 *)v16[0] != &v17 )
          j_j___libc_free_0(v16[0], v17 + 1);
      }
    }
    else
    {
      v14 = (char *)sub_BD5D20(a2);
      sub_B2F7A0(a1, v14, v15, 0, (__int64)byte_3F871B3, 0);
    }
  }
  else
  {
    v4 = sub_ED08F0(a2);
    v5 = *(_BYTE *)(a2 + 32);
    v6 = v4;
    v8 = v7;
    v9 = (char *)sub_BD5D20(a2);
    sub_B2F7A0(a1, v9, v10, v5 & 0xF, v6, v8);
  }
  return a1;
}
