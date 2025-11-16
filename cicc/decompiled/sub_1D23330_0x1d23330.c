// Function: sub_1D23330
// Address: 0x1d23330
//
__int64 __fastcall sub_1D23330(__int64 a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 v6; // rax
  char v7; // dl
  __int64 v8; // rax
  unsigned __int8 v9; // dl
  unsigned int v10; // ecx
  __int64 result; // rax
  unsigned int v12; // [rsp+8h] [rbp-58h]
  _BYTE v13[8]; // [rsp+10h] [rbp-50h] BYREF
  __int64 v14; // [rsp+18h] [rbp-48h]
  unsigned __int64 v15; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v16; // [rsp+28h] [rbp-38h]

  v6 = *(_QWORD *)(a2 + 40) + 16LL * (unsigned int)a3;
  v7 = *(_BYTE *)v6;
  v8 = *(_QWORD *)(v6 + 8);
  v13[0] = v7;
  v14 = v8;
  if ( v7 )
  {
    v9 = v7 - 14;
    if ( v9 <= 0x5Fu )
    {
      v10 = word_42E7700[v9];
      v16 = v10;
      if ( v10 <= 0x40 )
      {
LABEL_4:
        v15 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v10;
        goto LABEL_7;
      }
      goto LABEL_12;
    }
LABEL_6:
    v16 = 1;
    v15 = 1;
    goto LABEL_7;
  }
  if ( !(unsigned __int8)sub_1F58D20(v13) )
    goto LABEL_6;
  v10 = sub_1F58D30(v13);
  v16 = v10;
  if ( v10 <= 0x40 )
    goto LABEL_4;
LABEL_12:
  sub_16A4EF0((__int64)&v15, -1, 1);
LABEL_7:
  result = sub_1D210A0(a1, a2, a3, (__int64)&v15, a4);
  if ( v16 > 0x40 )
  {
    if ( v15 )
    {
      v12 = result;
      j_j___libc_free_0_0(v15);
      return v12;
    }
  }
  return result;
}
