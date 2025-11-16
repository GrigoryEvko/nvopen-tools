// Function: sub_1079CD0
// Address: 0x1079cd0
//
void *__fastcall sub_1079CD0(__int64 a1, char a2)
{
  void *result; // rax

  result = *(void **)a1;
  if ( !*(_QWORD *)a1 && (*(_BYTE *)(a1 + 9) & 0x70) == 0x20 && *(char *)(a1 + 8) >= 0 )
  {
    *(_BYTE *)(a1 + 8) = *(_BYTE *)(a1 + 8) & 0xF7 | (8 * (((*(_BYTE *)(a1 + 8) & 8) != 0) | a2 & 1));
    result = sub_E807D0(*(_QWORD *)(a1 + 24));
    *(_QWORD *)a1 = result;
  }
  return result;
}
