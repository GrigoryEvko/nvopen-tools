// Function: sub_B82100
// Address: 0xb82100
//
__int64 __fastcall sub_B82100(__int64 a1, __int64 a2)
{
  __int64 v3; // r12
  __int64 v4; // rbx
  __int64 v5; // r12
  __int64 v6; // rdi
  __int64 v7; // rsi

  v3 = *(unsigned int *)(a1 + 608);
  v4 = *(_QWORD *)(a1 + 600);
  *(_QWORD *)a1 = off_49DAB70;
  *(_QWORD *)(a1 + 176) = &unk_49DAC30;
  v5 = v4 + 16 * v3;
  if ( v4 != v5 )
  {
    do
    {
      v6 = *(_QWORD *)(v4 + 8);
      if ( v6 )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v6 + 8LL))(v6);
      v4 += 16;
    }
    while ( v5 != v4 );
    v5 = *(_QWORD *)(a1 + 600);
  }
  if ( v5 != a1 + 616 )
    _libc_free(v5, a2);
  v7 = 16LL * *(unsigned int *)(a1 + 592);
  sub_C7D6A0(*(_QWORD *)(a1 + 576), v7, 8);
  sub_B81E70(a1 + 176, v7);
  sub_BB9100(a1);
  return j_j___libc_free_0(a1, 632);
}
