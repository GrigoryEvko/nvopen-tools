// Function: sub_2FA5990
// Address: 0x2fa5990
//
_BYTE *__fastcall sub_2FA5990(_QWORD *a1, unsigned int **a2, __int64 a3, __int64 a4, int a5, _BYTE *a6)
{
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  _BYTE *v12; // rax
  bool v13; // zf
  _BYTE *result; // rax
  _BYTE *v16[4]; // [rsp+10h] [rbp-80h] BYREF
  _BYTE *v17; // [rsp+30h] [rbp-60h] BYREF
  __int16 v18; // [rsp+50h] [rbp-40h]

  v9 = sub_BCB2D0(a1);
  v16[0] = (_BYTE *)sub_ACD640(v9, 0, 0);
  v10 = sub_BCB2D0(a1);
  v16[1] = (_BYTE *)sub_ACD640(v10, 0, 0);
  v11 = sub_BCB2D0(a1);
  v12 = (_BYTE *)sub_ACD640(v11, a5, 0);
  v13 = *a6 == 0;
  v16[2] = v12;
  v18 = 257;
  if ( !v13 )
  {
    v17 = a6;
    LOBYTE(v18) = 3;
  }
  result = (_BYTE *)sub_921130(a2, a3, a4, v16, 3, (__int64)&v17, 0);
  if ( *result != 63 )
    return 0;
  return result;
}
