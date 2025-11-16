// Function: sub_EAFF30
// Address: 0xeaff30
//
__int64 __fastcall sub_EAFF30(_QWORD *a1, __int64 a2, __int64 a3)
{
  unsigned int v5; // r13d
  __int64 v6; // rsi
  __int64 i; // rax
  __int64 v8; // rdx
  _QWORD v10[4]; // [rsp+0h] [rbp-80h] BYREF
  __int16 v11; // [rsp+20h] [rbp-60h]
  _QWORD v12[4]; // [rsp+30h] [rbp-50h] BYREF
  __int16 v13; // [rsp+50h] [rbp-30h]

  v5 = sub_ECE000(a1);
  if ( (_BYTE)v5 )
    return v5;
  v6 = a1[47];
  if ( a1[46] != v6 )
  {
    for ( i = a1[41]; *(_QWORD *)(*(_QWORD *)(v6 - 8) + 24LL) != (i - a1[40]) >> 3; *(_QWORD *)((char *)a1 + 308) = v8 )
    {
      v8 = *(_QWORD *)(i - 8);
      i -= 8;
      a1[41] = i;
    }
    sub_EAFEB0((__int64)a1);
    return v5;
  }
  v11 = 1283;
  v10[0] = "unexpected '";
  v12[0] = v10;
  v13 = 770;
  v10[2] = a2;
  v10[3] = a3;
  v12[2] = "' in file, no current macro definition";
  return (unsigned int)sub_ECE0E0(a1, v12, 0, 0);
}
