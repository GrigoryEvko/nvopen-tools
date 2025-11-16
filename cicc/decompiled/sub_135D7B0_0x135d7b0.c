// Function: sub_135D7B0
// Address: 0x135d7b0
//
__int64 __fastcall sub_135D7B0(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  *(_BYTE *)(a2 + 160) = 1;
  sub_1636A40(a2, &unk_4F9D764);
  sub_1636A40(a2, &unk_4F9E06C);
  sub_1636A40(a2, &unk_4F9B6E8);
  result = *(unsigned int *)(a2 + 152);
  if ( (unsigned int)result >= *(_DWORD *)(a2 + 156) )
  {
    sub_16CD150(a2 + 144, a2 + 160, 0, 8);
    result = *(unsigned int *)(a2 + 152);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 144) + 8 * result) = &unk_4F99CC4;
  ++*(_DWORD *)(a2 + 152);
  return result;
}
