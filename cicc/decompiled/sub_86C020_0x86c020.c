// Function: sub_86C020
// Address: 0x86c020
//
__int64 __fastcall sub_86C020(__int64 a1)
{
  __int64 v1; // rbx
  __int64 result; // rax

  v1 = unk_4D03B98 + 176LL * unk_4D03B90;
  result = *(_QWORD *)v1 & 0x100FFFFFFFFLL;
  if ( result == 0x10000000000LL )
  {
    sub_86BFC0(*(_QWORD *)(v1 + 128));
    sub_733780(0x15u, a1, 0, 2, 0);
    result = qword_4F06BC0;
    *(_BYTE *)(v1 + 5) &= ~1u;
    *(_QWORD *)(v1 + 128) = result;
  }
  return result;
}
