// Function: sub_ED2A00
// Address: 0xed2a00
//
__int64 *__fastcall sub_ED2A00(__int64 *a1, __int64 a2, char a3)
{
  _BYTE *v3; // r14
  __int64 v4; // rdx
  __int64 v5; // r15
  char *v6; // rax
  __int64 v7; // rdx
  __int64 v9; // rax
  _BYTE *v10; // rsi
  __int64 v11; // rdx
  char *v12; // rax
  __int64 v13; // rdx
  int v14; // [rsp+Ch] [rbp-64h]
  _QWORD v15[2]; // [rsp+10h] [rbp-60h] BYREF
  __int64 v16; // [rsp+20h] [rbp-50h] BYREF
  char v17; // [rsp+30h] [rbp-40h]

  if ( a3 )
  {
    v9 = sub_ED29A0(a2);
    sub_ED0A20((__int64)v15, v9);
    if ( v17 )
    {
      v10 = (_BYTE *)v15[0];
      v11 = v15[1];
      *a1 = (__int64)(a1 + 2);
      sub_ED0570(a1, v10, (__int64)&v10[v11]);
      if ( v17 )
      {
        v17 = 0;
        if ( (__int64 *)v15[0] != &v16 )
          j_j___libc_free_0(v15[0], v16 + 1);
      }
    }
    else
    {
      v12 = (char *)sub_BD5D20(a2);
      sub_ED1420(a1, v12, v13, 0, byte_3F871B3, 0);
    }
  }
  else
  {
    v3 = (_BYTE *)sub_ED08F0(a2);
    v5 = v4;
    v14 = *(_BYTE *)(a2 + 32) & 0xF;
    v6 = (char *)sub_BD5D20(a2);
    sub_ED1420(a1, v6, v7, v14, v3, v5);
  }
  return a1;
}
