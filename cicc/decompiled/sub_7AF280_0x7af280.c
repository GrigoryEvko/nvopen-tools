// Function: sub_7AF280
// Address: 0x7af280
//
_DWORD *__fastcall sub_7AF280(__int64 a1, int a2)
{
  _DWORD *result; // rax
  int v3; // edx

  result = &dword_4D04930;
  if ( dword_4D04930 )
  {
    if ( HIDWORD(qword_4F077B4) | unk_4D04954 )
      fputc(35, qword_4D04928);
    else
      fwrite("#line", 1u, 5u, qword_4D04928);
    v3 = *(_DWORD *)(unk_4F064B0 + 40LL);
    if ( a2 )
    {
      fprintf(qword_4D04928, " %lu \"", (unsigned int)(v3 + 1));
      sub_723850(*(char **)(unk_4F064B0 + 8LL), qword_4D04928, unk_4D04954 == 0, 1);
      fputc(34, qword_4D04928);
      fputc(10, qword_4D04928);
      dword_4D03CEC = unk_4F06468 + 1;
      return (_DWORD *)(unsigned int)(unk_4F06468 + 1);
    }
    else
    {
      fprintf(qword_4D04928, " %lu \"", (unsigned int)(unk_4F0647C + v3 - unk_4F06468));
      sub_723850(*(char **)(unk_4F064B0 + 8LL), qword_4D04928, unk_4D04954 == 0, 1);
      fputc(34, qword_4D04928);
      fputc(10, qword_4D04928);
      dword_4D03CEC = unk_4F0647C;
      return &dword_4D03CEC;
    }
  }
  return result;
}
