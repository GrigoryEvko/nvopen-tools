// Function: sub_39FAD40
// Address: 0x39fad40
//
int __fastcall sub_39FAD40(void (*a1)(void *))
{
  void *v1; // rdx

  if ( &qword_4A427C0 )
    v1 = (void *)qword_4A427C0;
  else
    v1 = 0;
  return __cxa_atexit(a1, 0, v1);
}
