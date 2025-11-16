// Function: sub_2FE3C30
// Address: 0x2fe3c30
//
__int64 __fastcall sub_2FE3C30(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 (*v5)(); // r15
  __int64 v6; // rdx
  __int64 v7; // rbx
  unsigned int v8; // eax
  __int64 v9; // rdx
  unsigned int v11; // [rsp+8h] [rbp-38h]

  v5 = *(__int64 (**)())(*(_QWORD *)a1 + 1432LL);
  v11 = sub_350FA40(a3, a4);
  v7 = v6;
  v8 = sub_350FA40(a2, a4);
  if ( v5 == sub_2FE34A0 )
    return 0;
  else
    return ((__int64 (__fastcall *)(__int64, _QWORD, __int64, _QWORD, __int64))v5)(a1, v8, v9, v11, v7);
}
