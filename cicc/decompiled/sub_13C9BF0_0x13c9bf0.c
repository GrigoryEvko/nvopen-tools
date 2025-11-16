// Function: sub_13C9BF0
// Address: 0x13c9bf0
//
__int64 __fastcall sub_13C9BF0(__int64 a1)
{
  __int64 v1; // rax
  __int64 result; // rax
  int v3; // r12d

  if ( (unsigned int)sub_16AF4C0(&dword_4F98F48, 1, 0) )
  {
    do
    {
      v3 = dword_4F98F48;
      result = sub_16AF4B0();
    }
    while ( v3 != 2 );
  }
  else
  {
    sub_14CAFD0(a1);
    sub_13FBE20(a1);
    sub_15CD350(a1);
    sub_1458320(a1);
    v1 = sub_22077B0(80);
    if ( v1 )
    {
      *(_QWORD *)(v1 + 8) = 24;
      *(_QWORD *)v1 = "Induction Variable Users";
      *(_QWORD *)(v1 + 16) = "iv-users";
      *(_QWORD *)(v1 + 24) = 8;
      *(_QWORD *)(v1 + 32) = &unk_4F98F4C;
      *(_WORD *)(v1 + 40) = 256;
      *(_BYTE *)(v1 + 42) = 0;
      *(_QWORD *)(v1 + 48) = 0;
      *(_QWORD *)(v1 + 56) = 0;
      *(_QWORD *)(v1 + 64) = 0;
      *(_QWORD *)(v1 + 72) = sub_13CA060;
    }
    sub_163A800(a1, v1, 1);
    result = sub_16AF4B0();
    dword_4F98F48 = 2;
  }
  return result;
}
