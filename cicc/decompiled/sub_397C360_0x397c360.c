// Function: sub_397C360
// Address: 0x397c360
//
__int64 __fastcall sub_397C360(_QWORD *a1, __int64 a2, unsigned int a3)
{
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // r15
  unsigned int *v7; // r14
  __int64 v9; // r14
  __int64 (__fastcall *v10)(__int64, _QWORD, _QWORD); // r15
  unsigned int v11; // eax

  if ( a2 )
  {
    v4 = sub_396DD80((__int64)a1);
    v5 = (*(__int64 (__fastcall **)(__int64, __int64, _QWORD, _QWORD, _QWORD, _QWORD))(*(_QWORD *)v4 + 80LL))(
           v4,
           a2,
           a3,
           a1[29],
           a1[34],
           a1[32]);
    v6 = a1[32];
    v7 = (unsigned int *)v5;
    sub_397C300((__int64)a1, a3);
    return sub_38DDD30(v6, v7);
  }
  else
  {
    v9 = a1[32];
    v10 = *(__int64 (__fastcall **)(__int64, _QWORD, _QWORD))(*(_QWORD *)v9 + 424LL);
    v11 = sub_397C300((__int64)a1, a3);
    return v10(v9, 0, v11);
  }
}
