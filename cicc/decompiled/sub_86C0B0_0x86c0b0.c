// Function: sub_86C0B0
// Address: 0x86c0b0
//
void __fastcall sub_86C0B0(__int64 *a1, __int64 a2, __int64 a3)
{
  char v3; // al
  __int64 v4; // rax

  while ( a1 )
  {
    v3 = *((_BYTE *)a1 + 32);
    if ( v3 == 2 || v3 == 3 )
    {
      v4 = a1[5];
      if ( *(_QWORD *)(v4 + 80) == a2 )
        *(_QWORD *)(v4 + 80) = a3;
    }
    else if ( !v3 && !a1[8] && (a1[9] & 1) == 0 )
    {
      a1 = (__int64 *)a1[5];
    }
    a1 = (__int64 *)*a1;
  }
}
