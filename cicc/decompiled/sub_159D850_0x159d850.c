// Function: sub_159D850
// Address: 0x159d850
//
__int64 __fastcall sub_159D850(__int64 a1)
{
  __int64 i; // rdi
  __int64 v3; // rax

  switch ( *(_BYTE *)(a1 + 16) )
  {
    case 0:
    case 1:
    case 2:
    case 3:
      sub_41A076();
    case 4:
      sub_1594F50(a1);
      break;
    case 5:
      sub_159D7A0((__int64 ***)a1);
      break;
    case 6:
      sub_15979C0((__int64 **)a1);
      break;
    case 7:
      sub_1597CA0((__int64 **)a1);
      break;
    case 8:
      sub_1597F80((__int64 **)a1);
      break;
    case 9:
      sub_1594C40((__int64 *)a1);
      break;
    case 0xA:
      sub_1594A60((__int64 *)a1);
      break;
    case 0xB:
    case 0xC:
      sub_1595970(a1);
      break;
    case 0xD:
      def_1582D86();
    case 0xE:
      def_1582D86();
    case 0xF:
      sub_1594B80((__int64 *)a1);
      break;
    case 0x10:
      def_1582D86();
  }
  for ( i = *(_QWORD *)(a1 + 8); i; i = *(_QWORD *)(a1 + 8) )
  {
    v3 = sub_1648700(i);
    sub_159D850(v3);
  }
  sub_164BE60(a1);
  return sub_1648B90(a1);
}
