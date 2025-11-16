// Function: sub_2FEB980
// Address: 0x2feb980
//
__int64 __fastcall sub_2FEB980(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        unsigned int a6,
        unsigned __int8 a7,
        unsigned __int16 a8,
        _DWORD *a9)
{
  __int64 v11; // rbx
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 result; // rax
  __int64 (*v15)(); // r10
  __int64 v16; // [rsp+10h] [rbp-50h] BYREF
  __int64 v17; // [rsp+18h] [rbp-48h]
  __int64 v18; // [rsp+20h] [rbp-40h]
  __int64 v19; // [rsp+28h] [rbp-38h]

  v16 = a4;
  v17 = a5;
  v11 = sub_3007410(&v16, a2);
  if ( (_WORD)v16 )
  {
    if ( (_WORD)v16 == 1 || (unsigned __int16)(v16 - 504) <= 7u )
      BUG();
    v12 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v16 - 16];
  }
  else
  {
    v12 = sub_3007260(&v16);
    v18 = v12;
    v19 = v13;
  }
  if ( v12 && (unsigned __int8)sub_AE5020(a3, v11) > a7 )
  {
    v15 = *(__int64 (**)())(*(_QWORD *)a1 + 808LL);
    result = 0;
    if ( v15 != sub_2D56600 )
      return ((__int64 (__fastcall *)(__int64, _QWORD, __int64, _QWORD, _QWORD, _QWORD, _DWORD *))v15)(
               a1,
               (unsigned int)v16,
               v17,
               a6,
               a7,
               a8,
               a9);
  }
  else
  {
    if ( a9 )
      *a9 = 1;
    return 1;
  }
  return result;
}
