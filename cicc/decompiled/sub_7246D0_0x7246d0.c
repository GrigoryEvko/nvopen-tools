// Function: sub_7246D0
// Address: 0x7246d0
//
_BYTE *__fastcall sub_7246D0(__int64 a1)
{
  char *v1; // rax
  char v2; // si
  _BYTE *result; // rax
  char v4; // dl
  _QWORD *v5; // rax

  if ( dword_4F07270[0] == unk_4F073B8 )
  {
    v5 = (_QWORD *)(dword_4F07988 + sub_822F50(dword_4F07270[0], a1 + dword_4F0798C));
    if ( !unk_4D03FE8 )
      *v5++ = 0;
    *v5 = 0;
    result = v5 + 2;
    *(result - 8) = (2 * (unk_4D03FE8 == 0)) | (8 * (unk_4F06CFC & 1)) | 1;
  }
  else
  {
    v1 = (char *)(dword_4F07980 + sub_822F50(dword_4F07270[0], a1 + dword_4F07984));
    v2 = *v1;
    *v1 &= ~1u;
    result = v1 + 8;
    v4 = v2 & 0x88 | (2 * (unk_4D03FE8 == 0));
    *(result - 8) = v4;
    *(result - 8) = (8 * (unk_4F06CFC & 1)) | v4 & 0x77;
  }
  return result;
}
