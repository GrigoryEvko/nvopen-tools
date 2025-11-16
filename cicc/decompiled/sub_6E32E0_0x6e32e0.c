// Function: sub_6E32E0
// Address: 0x6e32e0
//
void __fastcall sub_6E32E0(__int64 a1, _QWORD *a2)
{
  __int64 i; // rbx

  for ( i = a1; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  if ( *(char *)(i + 168) < 0 || dword_4D047EC && (unsigned int)sub_8D4070(i) )
    *a2 = 0;
  else
    *a2 *= *(_QWORD *)(i + 176);
}
