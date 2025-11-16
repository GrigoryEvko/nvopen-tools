// Function: sub_CA83A0
// Address: 0xca83a0
//
void __fastcall sub_CA83A0(__int64 a1)
{
  char *v1; // rsi
  char *v2; // rax

  v1 = *(char **)(a1 + 40);
  if ( v1 != *(char **)(a1 + 48) && *v1 == 35 )
  {
    while ( 1 )
    {
      v2 = sub_CA6050(a1, v1);
      v1 = v2;
      if ( v2 == *(char **)(a1 + 40) )
        break;
      ++*(_DWORD *)(a1 + 60);
      *(_QWORD *)(a1 + 40) = v2;
    }
  }
}
