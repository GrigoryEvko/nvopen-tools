// Function: sub_23A2670
// Address: 0x23a2670
//
void __fastcall sub_23A2670(unsigned __int64 *a1, __int64 a2)
{
  _QWORD *v2; // rax
  unsigned __int64 v3; // rbx
  unsigned __int64 v4; // [rsp+8h] [rbp-328h] BYREF
  _BYTE v5[800]; // [rsp+10h] [rbp-320h] BYREF

  sub_234B220((__int64)v5, a2);
  v2 = (_QWORD *)sub_22077B0(0x300u);
  v3 = (unsigned __int64)v2;
  if ( v2 )
  {
    *v2 = &unk_4A0E7F8;
    sub_234B220((__int64)(v2 + 1), (__int64)v5);
  }
  v4 = v3;
  sub_23A2230(a1, &v4);
  if ( v4 )
    (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v4 + 8LL))(v4);
  sub_233AAF0((__int64)v5);
}
