// Function: sub_39A0350
// Address: 0x39a0350
//
void __fastcall sub_39A0350(__int64 *a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 *v3; // rbx
  __int64 *v4; // r13
  __int64 v5; // rsi
  __int64 v7; // rsi
  _QWORD *v8; // rdi

  v7 = *a1;
  v8 = a1 + 14;
  if ( v8[4] == v8[5] )
  {
    nullsub_2031();
  }
  else
  {
    v2 = v7;
    (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(v2 + 256) + 160LL))(*(_QWORD *)(v2 + 256), a2, 0);
    v3 = (__int64 *)v8[4];
    v4 = (__int64 *)v8[5];
    while ( v4 != v3 )
    {
      v5 = *v3++;
      sub_397CF20(v2, v5);
    }
    sub_397C0C0(v2, 0, "EOM(3)");
  }
}
