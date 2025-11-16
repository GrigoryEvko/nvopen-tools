// Function: sub_88D3E0
// Address: 0x88d3e0
//
_DWORD *sub_88D3E0()
{
  unsigned __int64 v1; // rax
  int v2; // [rsp+4h] [rbp-2Ch] BYREF
  _QWORD v3[5]; // [rsp+8h] [rbp-28h] BYREF

  if ( dword_4F077C4 == 1 )
    byte_4F068B0[0] = (dword_4F06B98 == 0) + 1;
  else
    byte_4F068B0[0] = 0;
  byte_4B6DF90[0] = dword_4F06B98;
  unk_4F068AC = dword_4F077C4 != 1;
  unk_4F06AC9 = 9;
  unk_4F06AC8 = 10;
  if ( !unk_4F06AC0 )
    unk_4F06AC0 = unk_4F06A58;
  if ( unk_4F06AB8 )
  {
    sub_622920(unk_4F06895, v3, &v2);
    v1 = ~(-1LL << ((byte_4B6DF90[unk_4F06895] == 0) + LOBYTE(v3[0]) * (unsigned __int8)dword_4F06BA0 - 1));
    if ( unk_4F06A58 <= v1 )
      v1 = unk_4F06A58;
    if ( v1 < unk_4F06AB8 )
      unk_4F06AB8 = v1;
  }
  else
  {
    unk_4F06AB8 = unk_4F06A58;
  }
  if ( unk_4D04548 | unk_4D04558 )
  {
    sub_88CE10();
    sub_622920(byte_4F06A51[0], v3, &v2);
    unk_4F06AD4 = v3[0] * dword_4F06BA0 == 64;
  }
  dword_4F068B4 = 0;
  return &dword_4F068B4;
}
