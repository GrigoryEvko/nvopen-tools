// Function: sub_C64C70
// Address: 0xc64c70
//
void __fastcall sub_C64C70(pthread_mutex_t *a1)
{
  unsigned int v1; // eax

  if ( &_pthread_key_create )
  {
    v1 = pthread_mutex_lock(a1);
    if ( v1 )
      sub_4264C5(v1);
  }
}
