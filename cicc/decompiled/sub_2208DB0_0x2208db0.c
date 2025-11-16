// Function: sub_2208DB0
// Address: 0x2208db0
//
void __fastcall sub_2208DB0(pthread_mutex_t **a1)
{
  _QWORD *v1; // rax

  if ( &_pthread_key_create )
  {
    if ( pthread_mutex_unlock(*a1) )
    {
      v1 = (_QWORD *)sub_2252770(8);
      *v1 = off_4A046B8;
      sub_2253480(v1, &`typeinfo for'__gnu_cxx::__concurrence_unlock_error, sub_2208D70);
    }
  }
}
