// Function: sub_2647660
// Address: 0x2647660
//
void __fastcall sub_2647660(unsigned __int64 *a1, _QWORD *a2)
{
  unsigned __int64 v3; // rsi
  __int64 v4; // rax

  v3 = a1[1];
  if ( v3 != a1[2] )
  {
    if ( v3 )
    {
      *(_QWORD *)v3 = *a2;
      v4 = a2[1];
      *(_QWORD *)(v3 + 8) = v4;
      if ( v4 )
      {
        if ( &_pthread_key_create )
        {
          _InterlockedAdd((volatile signed __int32 *)(v4 + 8), 1u);
          v3 = a1[1];
          goto LABEL_7;
        }
        ++*(_DWORD *)(v4 + 8);
      }
      v3 = a1[1];
    }
LABEL_7:
    a1[1] = v3 + 16;
    return;
  }
  sub_2647470(a1, (char *)v3, a2);
}
