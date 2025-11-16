// Function: sub_25D6090
// Address: 0x25d6090
//
_QWORD *__fastcall sub_25D6090(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  _QWORD *v7; // rax
  _QWORD *v8; // rbx
  _QWORD *v10; // rax

  if ( qword_502E468[9] | qword_4FF0750 )
  {
    v7 = (_QWORD *)sub_22077B0(0x40u);
    v8 = v7;
    if ( v7 )
    {
      v7[1] = a2;
      v7[2] = a3;
      v7[3] = a4;
      v7[4] = a5;
      v7[5] = 0;
      v7[6] = 0;
      *v7 = &unk_4A1F2E8;
      v7[7] = 0x2800000000LL;
      if ( (qword_502E468[9] == 0) == (qword_4FF0750 == 0) )
        sub_C64ED0("Pass only one of: -thinlto-pgo-ctx-prof or -thinlto-workload-def", 1u);
      if ( qword_502E468[9] )
        sub_25D56C0((__int64)v7);
      else
        sub_25CF4F0((__int64)v7);
    }
    *a1 = v8;
  }
  else
  {
    v10 = (_QWORD *)sub_22077B0(0x28u);
    if ( v10 )
    {
      v10[1] = a2;
      v10[2] = a3;
      v10[3] = a4;
      *v10 = &unk_4A1F310;
      v10[4] = a5;
    }
    *a1 = v10;
  }
  return a1;
}
