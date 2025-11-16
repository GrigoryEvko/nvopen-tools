// Function: sub_2C27AE0
// Address: 0x2c27ae0
//
_QWORD *__fastcall sub_2C27AE0(_QWORD *a1, char a2, __int64 *a3, __int64 a4, int a5, char a6, __int64 *a7, void **a8)
{
  __int64 v9; // rax
  __int64 v11; // rax
  _QWORD *v12; // r12
  unsigned __int64 *v13; // rdx
  __int64 v14; // rax
  unsigned __int64 v15; // rsi
  __int64 *v17; // rdx
  __int64 v19; // [rsp+10h] [rbp-40h] BYREF
  __int64 v20[7]; // [rsp+18h] [rbp-38h] BYREF

  v9 = *a7;
  if ( a6 )
  {
    v20[0] = *a7;
    if ( v9 )
      sub_2C25AB0(v20);
    v11 = sub_22077B0(0xC8u);
    v12 = (_QWORD *)v11;
    if ( v11 )
      sub_2C1AF80(v11, a2, a3, a4, a5, v20, a8);
    if ( *a1 )
    {
      v13 = (unsigned __int64 *)a1[1];
      v12[10] = *a1;
      v14 = v12[3];
      v15 = *v13;
      v12[4] = v13;
      v15 &= 0xFFFFFFFFFFFFFFF8LL;
      v12[3] = v15 | v14 & 7;
      *(_QWORD *)(v15 + 8) = v12 + 3;
      *v13 = *v13 & 7 | (unsigned __int64)(v12 + 3);
    }
    sub_9C6650(v20);
  }
  else
  {
    v19 = *a7;
    if ( v9 )
    {
      sub_2C25AB0(&v19);
      v20[0] = v19;
      if ( v19 )
        sub_2C25AB0(v20);
    }
    else
    {
      v20[0] = 0;
    }
    v17 = 0;
    if ( 8 * a4 )
      v17 = a3;
    v12 = (_QWORD *)sub_2AAFFE0(a1, a2, v17, a4, v20, a8);
    sub_9C6650(v20);
    sub_9C6650(&v19);
  }
  return v12;
}
