// Function: sub_2450D10
// Address: 0x2450d10
//
_BYTE *__fastcall sub_2450D10(_QWORD ***a1, void *a2, unsigned __int64 a3)
{
  _BYTE *v4; // r12
  _QWORD *v6; // r15
  char v7; // al
  unsigned __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // [rsp+8h] [rbp-78h]
  __int64 v14; // [rsp+18h] [rbp-68h]
  _QWORD v15[4]; // [rsp+20h] [rbp-60h] BYREF
  __int16 v16; // [rsp+40h] [rbp-40h]

  v4 = sub_BA8CD0((__int64)*a1, (__int64)a2, a3, 0);
  if ( !v4 )
  {
    v6 = (_QWORD *)sub_BCB2E0(**a1);
    v15[0] = a2;
    v13 = sub_AD6530((__int64)v6, (__int64)a2);
    v16 = 261;
    v15[1] = a3;
    BYTE4(v14) = 0;
    v4 = sub_BD2C40(88, unk_3F0FAE8);
    if ( v4 )
      sub_B30000((__int64)v4, (__int64)*a1, v6, 0, 3, v13, (__int64)v15, 0, 0, v14, 0);
    v7 = v4[32] & 0xCF | 0x10;
    v4[32] = v7;
    if ( (v7 & 0xF) != 9 )
      v4[33] |= 0x40u;
    v8 = *((unsigned int *)a1 + 25);
    if ( (unsigned int)v8 > 8 || (v9 = 292, !_bittest64(&v9, v8)) )
    {
      v10 = sub_BAA410((__int64)*a1, a2, a3);
      sub_B2F990((__int64)v4, v10, v11, v12);
    }
  }
  return v4;
}
