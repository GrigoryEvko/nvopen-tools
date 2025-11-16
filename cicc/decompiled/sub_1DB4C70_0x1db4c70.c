// Function: sub_1DB4C70
// Address: 0x1db4c70
//
void __fastcall sub_1DB4C70(__int64 a1)
{
  _QWORD *v1; // r13
  __int64 i; // rbx
  unsigned __int64 *v3; // rsi

  v1 = (_QWORD *)(a1 + 104);
  for ( i = *(_QWORD *)(a1 + 104); i; *v1 = i )
  {
    while ( *(_DWORD *)(i + 8) )
    {
      v1 = (_QWORD *)(i + 104);
      i = *(_QWORD *)(i + 104);
      if ( !i )
        return;
    }
    do
    {
      v3 = (unsigned __int64 *)i;
      i = *(_QWORD *)(i + 104);
      sub_1DB4BE0(a1, v3);
    }
    while ( i && !*(_DWORD *)(i + 8) );
  }
}
