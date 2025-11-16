// Function: sub_398B390
// Address: 0x398b390
//
__int64 __fastcall sub_398B390(__int64 a1, __int64 a2)
{
  __int64 v3; // rdx
  __int64 v4; // r13
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // r15
  __int64 v8; // rdi
  __int64 v9; // rbx
  __int64 v10; // rdx
  __int64 v11; // r14

  sub_397C0C0(*(_QWORD *)(a1 + 8), *(unsigned __int16 *)(a2 + 2), 0);
  sub_397C0C0(*(_QWORD *)(a1 + 8), *(unsigned int *)(a2 + 24), 0);
  v3 = *(unsigned int *)(a2 + 8);
  v4 = *(_QWORD *)(a2 - 8 * v3);
  if ( v4 )
  {
    v5 = sub_161E970(*(_QWORD *)(a2 - 8LL * *(unsigned int *)(a2 + 8)));
    v7 = v6;
    v3 = *(unsigned int *)(a2 + 8);
    v4 = v5;
  }
  else
  {
    v7 = 0;
  }
  v8 = *(_QWORD *)(a2 + 8 * (1 - v3));
  if ( v8 )
  {
    v9 = sub_161E970(v8);
    v11 = v10;
    (*(void (__fastcall **)(_QWORD, __int64, __int64))(**(_QWORD **)(*(_QWORD *)(a1 + 8) + 256LL) + 400LL))(
      *(_QWORD *)(*(_QWORD *)(a1 + 8) + 256LL),
      v4,
      v7);
    if ( v11 )
    {
      sub_396F300(*(_QWORD *)(a1 + 8), 32);
      (*(void (__fastcall **)(_QWORD, __int64, __int64))(**(_QWORD **)(*(_QWORD *)(a1 + 8) + 256LL) + 400LL))(
        *(_QWORD *)(*(_QWORD *)(a1 + 8) + 256LL),
        v9,
        v11);
    }
  }
  else
  {
    (*(void (__fastcall **)(_QWORD, __int64, __int64))(**(_QWORD **)(*(_QWORD *)(a1 + 8) + 256LL) + 400LL))(
      *(_QWORD *)(*(_QWORD *)(a1 + 8) + 256LL),
      v4,
      v7);
  }
  return sub_396F300(*(_QWORD *)(a1 + 8), 0);
}
