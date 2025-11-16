// Function: sub_5D5250
// Address: 0x5d5250
//
int __fastcall sub_5D5250(__int64 a1)
{
  __int64 v2; // rdi
  int result; // eax

  if ( unk_4F072C8 != 1 )
    return sub_748000(a1, 1, &qword_4CF7CE0);
  v2 = *(_QWORD *)(a1 + 128);
  if ( !v2 || !(unsigned int)sub_8D2E30(v2) || !(unsigned int)sub_8D9600(*(_QWORD *)(a1 + 128), sub_5D3AF0, 19) )
    return sub_748000(a1, 1, &qword_4CF7CE0);
  result = putc(48, stream);
  ++dword_4CF7F40;
  return result;
}
