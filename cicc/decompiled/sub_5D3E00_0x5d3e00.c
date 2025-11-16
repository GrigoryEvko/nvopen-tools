// Function: sub_5D3E00
// Address: 0x5d3e00
//
int __fastcall sub_5D3E00(__int64 a1)
{
  int result; // eax

  if ( unk_4F072C8 != 1 || !(unsigned int)sub_8D2E30(a1) || (result = sub_8D9600(a1, sub_5D3AF0, 19)) == 0 )
  {
    putc(40, stream);
    ++dword_4CF7F40;
    sub_74A390(a1, 0, 0, 0, 0, &qword_4CF7CE0);
    sub_74D110(a1, 0, 0, &qword_4CF7CE0);
    result = putc(41, stream);
    ++dword_4CF7F40;
  }
  return result;
}
