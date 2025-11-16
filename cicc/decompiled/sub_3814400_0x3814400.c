// Function: sub_3814400
// Address: 0x3814400
//
__int64 __fastcall sub_3814400(__int64 a1, __int64 a2, unsigned int a3, __int64 a4)
{
  __int64 (__fastcall *v7)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v8; // rdx
  char v9[8]; // [rsp+0h] [rbp-50h] BYREF
  __int16 v10; // [rsp+8h] [rbp-48h]
  __int64 v11; // [rsp+10h] [rbp-40h]

  while ( 1 )
  {
    sub_2FE6CC0((__int64)v9, a1, a2, a3, a4);
    if ( !v9[0] )
      break;
    if ( v9[0] != 2 )
      BUG();
    v7 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)a1 + 592LL);
    if ( v7 == sub_2D56A50 )
    {
      sub_2FE6CC0((__int64)v9, a1, a2, a3, a4);
      LOWORD(a3) = v10;
      a4 = v11;
    }
    else
    {
      a3 = v7(a1, a2, a3, a4);
      a4 = v8;
    }
  }
  return a3;
}
