// Function: sub_30CB170
// Address: 0x30cb170
//
void __fastcall sub_30CB170(__int64 a1, const void *a2, size_t a3)
{
  _QWORD *v4; // rax
  __int64 v5; // r12
  __int64 *v6; // rax

  if ( (_BYTE)qword_502FA68 )
  {
    v4 = (_QWORD *)sub_BD5C60(a1);
    v5 = sub_A78730(v4, "inline-remark", 0xDu, a2, a3);
    v6 = (__int64 *)sub_BD5C60(a1);
    *(_QWORD *)(a1 + 72) = sub_A7B440((__int64 *)(a1 + 72), v6, -1, v5);
  }
}
