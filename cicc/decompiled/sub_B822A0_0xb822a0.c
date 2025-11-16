// Function: sub_B822A0
// Address: 0xb822a0
//
__int64 __fastcall sub_B822A0(__int64 a1, __int64 a2)
{
  __int64 v3; // r12
  __int64 v4; // rbx
  __int64 v5; // r12
  __int64 v6; // rdi
  __int64 v7; // rsi

  v3 = *(unsigned int *)(a1 + 432);
  v4 = *(_QWORD *)(a1 + 424);
  *(_QWORD *)(a1 - 176) = off_49DAB70;
  *(_QWORD *)a1 = &unk_49DAC30;
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
    v5 = *(_QWORD *)(a1 + 424);
  }
  if ( v5 != a1 + 440 )
    _libc_free(v5, a2);
  v7 = 16LL * *(unsigned int *)(a1 + 416);
  sub_C7D6A0(*(_QWORD *)(a1 + 400), v7, 8);
  sub_B81E70(a1, v7);
  return sub_BB9100(a1 - 176);
}
