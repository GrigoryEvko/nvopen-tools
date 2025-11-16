// Function: sub_16F7C20
// Address: 0x16f7c20
//
void __fastcall sub_16F7C20(__int64 a1)
{
  char *v1; // rsi
  char *v2; // rax

  v1 = *(char **)(a1 + 40);
  if ( *v1 == 35 )
  {
    while ( 1 )
    {
      v2 = sub_16F6380(a1, v1);
      v1 = v2;
      if ( v2 == *(char **)(a1 + 40) )
        break;
      ++*(_DWORD *)(a1 + 60);
      *(_QWORD *)(a1 + 40) = v2;
    }
  }
}
