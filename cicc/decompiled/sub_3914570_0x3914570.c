// Function: sub_3914570
// Address: 0x3914570
//
__int64 __fastcall sub_3914570(__int64 a1, char *a2, size_t a3, int a4)
{
  __int64 v6; // r13
  void *v8; // rdi

  v6 = *(_QWORD *)(a1 + 240);
  v8 = *(void **)(v6 + 24);
  if ( *(_QWORD *)(v6 + 16) - (_QWORD)v8 < a3 )
  {
    sub_16E7EE0(v6, a2, a3);
    v6 = *(_QWORD *)(a1 + 240);
  }
  else if ( a3 )
  {
    memcpy(v8, a2, a3);
    *(_QWORD *)(v6 + 24) += a3;
    v6 = *(_QWORD *)(a1 + 240);
  }
  return sub_16E8900(v6, a4 - (int)a3);
}
