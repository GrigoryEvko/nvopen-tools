// Function: sub_B821D0
// Address: 0xb821d0
//
__int64 __fastcall sub_B821D0(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 v4; // r12
  __int64 v5; // rbx
  __int64 v6; // r12
  __int64 v7; // rdi
  __int64 v8; // rsi

  v2 = a1 - 176;
  v4 = *(unsigned int *)(a1 + 432);
  v5 = *(_QWORD *)(a1 + 424);
  *(_QWORD *)(a1 - 176) = off_49DAB70;
  *(_QWORD *)a1 = &unk_49DAC30;
  v6 = v5 + 16 * v4;
  if ( v5 != v6 )
  {
    do
    {
      v7 = *(_QWORD *)(v5 + 8);
      if ( v7 )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v7 + 8LL))(v7);
      v5 += 16;
    }
    while ( v6 != v5 );
    v6 = *(_QWORD *)(a1 + 424);
  }
  if ( v6 != a1 + 440 )
    _libc_free(v6, a2);
  v8 = 16LL * *(unsigned int *)(a1 + 416);
  sub_C7D6A0(*(_QWORD *)(a1 + 400), v8, 8);
  sub_B81E70(a1, v8);
  sub_BB9100(v2);
  return j_j___libc_free_0(v2, 632);
}
