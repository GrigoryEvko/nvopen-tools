// Function: sub_222DFF0
// Address: 0x222dff0
//
void __fastcall sub_222DFF0(__int64 a1)
{
  unsigned __int64 v2; // rdi
  int v3; // eax
  unsigned __int64 v4; // rbx

  v2 = *(_QWORD *)(a1 + 40);
  if ( v2 )
  {
    while ( 1 )
    {
      if ( &_pthread_key_create )
      {
        if ( _InterlockedExchangeAdd((volatile signed __int32 *)(v2 + 20), 0xFFFFFFFF) )
          break;
      }
      else
      {
        v3 = *(_DWORD *)(v2 + 20);
        *(_DWORD *)(v2 + 20) = v3 - 1;
        if ( v3 )
          break;
      }
      v4 = *(_QWORD *)v2;
      j___libc_free_0(v2);
      if ( !v4 )
        break;
      v2 = v4;
    }
  }
  *(_QWORD *)(a1 + 40) = 0;
}
