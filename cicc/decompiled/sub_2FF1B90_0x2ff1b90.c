// Function: sub_2FF1B90
// Address: 0x2ff1b90
//
__int64 __fastcall sub_2FF1B90(__int64 a1)
{
  _QWORD *v1; // rsi
  __int64 (*v2)(); // rax

  if ( (_QWORD *(*)())qword_5026F68 != sub_2F42900 && (__int64 (*)())qword_5026F68 != sub_2FEDDD0 )
    sub_C64ED0("Must use fast (default) register allocator for unoptimized regalloc.", 1u);
  v1 = (_QWORD *)(*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 424LL))(a1, 0);
  sub_2FF0E80(a1, v1, 0);
  v2 = *(__int64 (**)())(*(_QWORD *)a1 + 344LL);
  if ( v2 != sub_2FEDAF0 )
    ((void (__fastcall *)(__int64))v2)(a1);
  return 1;
}
