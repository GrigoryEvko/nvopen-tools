// Function: sub_2FA58E0
// Address: 0x2fa58e0
//
_BYTE *__fastcall sub_2FA58E0(_QWORD *a1, unsigned int **a2, __int64 a3, __int64 a4, int a5, _BYTE *a6)
{
  __int64 v9; // rax
  __int64 v10; // rax
  _BYTE *v11; // rax
  bool v12; // zf
  _BYTE *result; // rax
  _BYTE *v15[2]; // [rsp+10h] [rbp-70h] BYREF
  _BYTE *v16; // [rsp+20h] [rbp-60h] BYREF
  __int16 v17; // [rsp+40h] [rbp-40h]

  v9 = sub_BCB2D0(a1);
  v15[0] = (_BYTE *)sub_ACD640(v9, 0, 0);
  v10 = sub_BCB2D0(a1);
  v11 = (_BYTE *)sub_ACD640(v10, a5, 0);
  v12 = *a6 == 0;
  v15[1] = v11;
  v17 = 257;
  if ( !v12 )
  {
    v16 = a6;
    LOBYTE(v17) = 3;
  }
  result = (_BYTE *)sub_921130(a2, a3, a4, v15, 2, (__int64)&v16, 0);
  if ( *result != 63 )
    return 0;
  return result;
}
