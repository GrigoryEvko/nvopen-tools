// Function: sub_67EA10
// Address: 0x67ea10
//
int __fastcall sub_67EA10(int a1, __int64 a2)
{
  int result; // eax
  char *v3; // rax

  result = unk_4D044E4;
  if ( !unk_4D044E4 )
  {
    v3 = sub_67C860(a1);
    fprintf(qword_4F07510, v3, qword_4F076F0, a2);
    return fputc(10, qword_4F07510);
  }
  return result;
}
