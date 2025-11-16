// Function: sub_88CE10
// Address: 0x88ce10
//
__int64 sub_88CE10()
{
  __int64 result; // rax

  if ( dword_4F06BA0 == 8 )
  {
    unk_4F06AD1 = byte_4F068B0[0];
    unk_4F06AD0 = 2;
  }
  unk_4F06ACF = sub_622A90(0x10u, 1);
  if ( unk_4F06ACF != 13 )
    unk_4F06ACE = sub_622A90(0x10u, 0);
  unk_4F06ACD = sub_622A90(0x20u, 1);
  if ( unk_4F06ACD != 13 )
    unk_4F06ACC = sub_622A90(0x20u, 0);
  result = sub_622A90(0x40u, 1);
  unk_4F06ACB = result;
  if ( (_BYTE)result != 13 )
  {
    result = sub_622A90(0x40u, 0);
    unk_4F06ACA = result;
  }
  return result;
}
