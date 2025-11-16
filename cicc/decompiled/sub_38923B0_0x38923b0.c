// Function: sub_38923B0
// Address: 0x38923b0
//
__int64 __fastcall sub_38923B0(__int64 a1, __int64 **a2, char a3)
{
  __int64 v3; // r14
  unsigned __int64 v4; // r15
  unsigned int v5; // r13d
  unsigned __int64 v6; // rsi
  unsigned __int64 v8; // rax
  const char *v9; // rax
  const char *v10; // rax
  unsigned __int64 v11; // [rsp+0h] [rbp-80h]
  __int64 v12; // [rsp+10h] [rbp-70h]
  __int64 *v14; // [rsp+28h] [rbp-58h] BYREF
  _QWORD v15[2]; // [rsp+30h] [rbp-50h] BYREF
  char v16; // [rsp+40h] [rbp-40h]
  char v17; // [rsp+41h] [rbp-3Fh]

  v3 = a1 + 8;
  v4 = *(_QWORD *)(a1 + 56);
  if ( *(_DWORD *)(a1 + 64) == 390 && (v5 = *(unsigned __int8 *)(a1 + 164), (_BYTE)v5) && *(_DWORD *)(a1 + 160) <= 0x40u )
  {
    v12 = *(_QWORD *)(a1 + 152);
    *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
    if ( !(unsigned __int8)sub_388AF10(a1, 17, "expected 'x' after element count") )
    {
      v8 = *(_QWORD *)(a1 + 56);
      v14 = 0;
      v11 = v8;
      v15[0] = "expected type";
      v17 = 1;
      v16 = 3;
      if ( !(unsigned __int8)sub_3891B00(a1, (__int64 *)&v14, (__int64)v15, 0) )
      {
        if ( !a3 )
        {
          if ( (unsigned __int8)sub_388AF10(a1, 7, "expected end of sequential type") )
            return v5;
          if ( sub_1643EC0((__int64)v14) )
          {
            v5 = 0;
            *a2 = sub_1645D80(v14, v12);
            return v5;
          }
          v17 = 1;
          v10 = "invalid array element type";
          goto LABEL_17;
        }
        if ( !(unsigned __int8)sub_388AF10(a1, 11, "expected end of sequential type") )
        {
          if ( !v12 )
          {
            v17 = 1;
            v9 = "zero element vector is illegal";
            goto LABEL_13;
          }
          if ( v12 != (unsigned int)v12 )
          {
            v17 = 1;
            v9 = "size too large for vector";
LABEL_13:
            v15[0] = v9;
            v16 = 3;
            return (unsigned int)sub_38814C0(v3, v4, (__int64)v15);
          }
          if ( (unsigned __int8)sub_1643F10((__int64)v14) )
          {
            v5 = 0;
            *a2 = sub_16463B0(v14, v12);
            return v5;
          }
          v17 = 1;
          v10 = "invalid vector element type";
LABEL_17:
          v15[0] = v10;
          v16 = 3;
          return (unsigned int)sub_38814C0(v3, v11, (__int64)v15);
        }
      }
    }
  }
  else
  {
    v6 = *(_QWORD *)(a1 + 56);
    v17 = 1;
    v16 = 3;
    v15[0] = "expected number in address space";
    return (unsigned int)sub_38814C0(a1 + 8, v6, (__int64)v15);
  }
  return v5;
}
