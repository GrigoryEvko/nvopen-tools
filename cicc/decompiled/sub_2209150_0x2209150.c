// Function: sub_2209150
// Address: 0x2209150
//
void __fastcall sub_2209150(volatile signed __int32 **a1)
{
  volatile signed __int32 *v1; // rbp
  volatile signed __int32 v2; // eax

  v1 = *a1;
  if ( *a1 != (volatile signed __int32 *)unk_4FD4F58 )
  {
    if ( !&_pthread_key_create )
    {
      v2 = (*v1)--;
      if ( v2 != 1 )
        return;
LABEL_6:
      sub_2208EC0(v1);
      j___libc_free_0((unsigned __int64)v1);
      return;
    }
    if ( _InterlockedExchangeAdd(v1, 0xFFFFFFFF) == 1 )
      goto LABEL_6;
  }
}
