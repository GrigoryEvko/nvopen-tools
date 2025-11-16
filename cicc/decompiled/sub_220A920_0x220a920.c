// Function: sub_220A920
// Address: 0x220a920
//
void sub_220A920()
{
  if ( !&_pthread_key_create )
  {
    if ( unk_4FD4F58 )
      return;
    goto LABEL_5;
  }
  pthread_once(&dword_4FD4F48, sub_220A8E0);
  if ( !unk_4FD4F58 )
LABEL_5:
    sub_220A8E0();
}
