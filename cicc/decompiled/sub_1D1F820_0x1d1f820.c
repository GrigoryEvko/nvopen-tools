// Function: sub_1D1F820
// Address: 0x1d1f820
//
void __fastcall sub_1D1F820(__int64 a1, __int64 a2, __int64 a3, unsigned __int64 *a4, int a5)
{
  __int64 v8; // rax
  char v9; // dl
  __int64 v10; // rax
  unsigned __int8 v11; // dl
  unsigned int v12; // ecx
  _BYTE v13[8]; // [rsp+10h] [rbp-50h] BYREF
  __int64 v14; // [rsp+18h] [rbp-48h]
  unsigned __int64 v15; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v16; // [rsp+28h] [rbp-38h]

  v8 = *(_QWORD *)(a2 + 40) + 16LL * (unsigned int)a3;
  v9 = *(_BYTE *)v8;
  v10 = *(_QWORD *)(v8 + 8);
  v13[0] = v9;
  v14 = v10;
  if ( v9 )
  {
    v11 = v9 - 14;
    if ( v11 <= 0x5Fu )
    {
      v12 = word_42E7700[v11];
      v16 = v12;
      if ( v12 <= 0x40 )
      {
LABEL_4:
        v15 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v12;
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
  v12 = sub_1F58D30(v13);
  v16 = v12;
  if ( v12 <= 0x40 )
    goto LABEL_4;
LABEL_12:
  sub_16A4EF0((__int64)&v15, -1, 1);
LABEL_7:
  sub_1D1B0D0(a1, a2, a3, a4, (__int64)&v15, a5);
  if ( v16 > 0x40 )
  {
    if ( v15 )
      j_j___libc_free_0_0(v15);
  }
}
