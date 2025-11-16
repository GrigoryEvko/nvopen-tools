// Function: sub_2FDDEF0
// Address: 0x2fddef0
//
__int64 __fastcall sub_2FDDEF0(__int64 a1, __int64 a2, char **a3)
{
  __int64 v4; // r12
  char *v5; // r14
  char *v6; // rbx
  __int64 v7; // rax
  char *v8; // r12
  __int64 result; // rax
  __int64 v10; // rax
  __int64 v11; // rax

  v4 = **(_QWORD **)(*((_QWORD *)*a3 + 3) + 32LL);
  if ( (unsigned __int8)sub_B2D620(v4, "target-features", 0xFu) )
  {
    v10 = sub_B2D7E0(v4, "target-features", 0xFu);
    sub_B2CDC0(a2, v10);
  }
  if ( (unsigned __int8)sub_B2D620(v4, "target-cpu", 0xAu) )
  {
    v11 = sub_B2D7E0(v4, "target-cpu", 0xAu);
    sub_B2CDC0(a2, v11);
  }
  v5 = a3[1];
  v6 = *a3;
  v7 = 0x6DB6DB6DB6DB6DB7LL * ((v5 - v6) >> 5);
  if ( v7 >> 2 > 0 )
  {
    v8 = &v6[896 * (v7 >> 2)];
    while ( 1 )
    {
      result = sub_B2D610(**(_QWORD **)(*((_QWORD *)v6 + 3) + 32LL), 41);
      if ( !(_BYTE)result )
        goto LABEL_12;
      result = sub_B2D610(**(_QWORD **)(*((_QWORD *)v6 + 31) + 32LL), 41);
      if ( !(_BYTE)result )
      {
        v6 += 224;
        goto LABEL_12;
      }
      result = sub_B2D610(**(_QWORD **)(*((_QWORD *)v6 + 59) + 32LL), 41);
      if ( !(_BYTE)result )
      {
        v6 += 448;
        goto LABEL_12;
      }
      result = sub_B2D610(**(_QWORD **)(*((_QWORD *)v6 + 87) + 32LL), 41);
      if ( !(_BYTE)result )
      {
        v6 += 672;
        goto LABEL_12;
      }
      v6 += 896;
      if ( v8 == v6 )
      {
        v7 = 0x6DB6DB6DB6DB6DB7LL * ((v5 - v6) >> 5);
        break;
      }
    }
  }
  if ( v7 == 2 )
    goto LABEL_21;
  if ( v7 == 3 )
  {
    result = sub_B2D610(**(_QWORD **)(*((_QWORD *)v6 + 3) + 32LL), 41);
    if ( !(_BYTE)result )
      goto LABEL_12;
    v6 += 224;
LABEL_21:
    result = sub_B2D610(**(_QWORD **)(*((_QWORD *)v6 + 3) + 32LL), 41);
    if ( !(_BYTE)result )
      goto LABEL_12;
    v6 += 224;
    goto LABEL_23;
  }
  if ( v7 != 1 )
    return sub_B2CD30(a2, 41);
LABEL_23:
  result = sub_B2D610(**(_QWORD **)(*((_QWORD *)v6 + 3) + 32LL), 41);
  if ( (_BYTE)result )
    return sub_B2CD30(a2, 41);
LABEL_12:
  if ( v5 == v6 )
    return sub_B2CD30(a2, 41);
  return result;
}
