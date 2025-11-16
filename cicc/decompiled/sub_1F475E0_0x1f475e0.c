// Function: sub_1F475E0
// Address: 0x1f475e0
//
__int64 __fastcall sub_1F475E0(__int64 a1, char a2)
{
  __int64 (*v2)(void); // rax
  int v4; // ebx
  __int64 (__fastcall *v5)(__int64, char); // rax

  if ( (unsigned int)sub_16AF4C0(&dword_4FCB750, 1, 0) )
  {
    do
    {
      v4 = dword_4FCB750;
      sub_16AF4B0();
    }
    while ( v4 != 2 );
    v2 = (__int64 (*)(void))unk_4FCB768;
    if ( (__int64 (*)())unk_4FCB768 != sub_1F448E0 )
      return v2();
  }
  else
  {
    if ( !unk_4FCB768 )
      unk_4FCB768 = qword_4FCB820;
    sub_16AF4B0();
    v2 = (__int64 (*)(void))unk_4FCB768;
    dword_4FCB750 = 2;
    if ( (__int64 (*)())unk_4FCB768 != sub_1F448E0 )
      return v2();
  }
  v5 = *(__int64 (__fastcall **)(__int64, char))(*(_QWORD *)a1 + 312LL);
  if ( v5 != sub_1F44D90 )
    return v5(a1, a2);
  if ( a2 )
    return sub_1EBDCD0();
  return sub_1EB6E00();
}
