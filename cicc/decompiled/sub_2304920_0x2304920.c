// Function: sub_2304920
// Address: 0x2304920
//
_QWORD *__fastcall sub_2304920(_QWORD *a1, __int64 a2)
{
  _QWORD *v2; // rax
  _BYTE v4[352]; // [rsp+0h] [rbp-440h] BYREF
  _BYTE v5[352]; // [rsp+160h] [rbp-2E0h] BYREF
  _BYTE v6[352]; // [rsp+2C0h] [rbp-180h] BYREF

  sub_30C8FF0(v4, a2 + 8);
  qmemcpy(v6, v4, sizeof(v6));
  v2 = (_QWORD *)sub_22077B0(0x168u);
  if ( v2 )
  {
    qmemcpy(v5, v6, sizeof(v5));
    *v2 = &unk_4A0B038;
    qmemcpy(v2 + 1, v5, 0x160u);
  }
  *a1 = v2;
  return a1;
}
