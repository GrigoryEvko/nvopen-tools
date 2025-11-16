// Function: sub_C64CA0
// Address: 0xc64ca0
//
void __fastcall sub_C64CA0(__int64 a1, __int64 a2)
{
  unsigned int v2; // eax

  if ( &_pthread_key_create )
  {
    v2 = pthread_mutex_lock(&stru_4F840A0);
    if ( v2 )
      sub_4264C5(v2);
  }
  qword_4F840E0 = a1;
  qword_4F840D8 = a2;
  if ( &_pthread_key_create )
    pthread_mutex_unlock(&stru_4F840A0);
}
