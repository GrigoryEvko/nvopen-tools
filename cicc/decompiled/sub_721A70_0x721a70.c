// Function: sub_721A70
// Address: 0x721a70
//
unsigned __int64 __fastcall sub_721A70(FILE *stream)
{
  unsigned __int64 v1; // rax
  unsigned __int64 v2; // r13

  v1 = ftell(stream);
  v2 = sub_721A20(v1);
  if ( fseek(stream, v2, 0) )
    sub_721090();
  return v2;
}
