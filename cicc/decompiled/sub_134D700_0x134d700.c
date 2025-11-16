// Function: sub_134D700
// Address: 0x134d700
//
__int64 __fastcall sub_134D700(__int64 a1)
{
  __int64 v1; // rax
  __int64 result; // rax
  int v3; // ebx
  int v4; // ebx

  if ( (unsigned int)sub_16AF4C0(&dword_4F96DB8, 1, 0) )
  {
    do
    {
      v3 = dword_4F96DB8;
      result = sub_16AF4B0();
      if ( v3 == 2 )
        break;
      v4 = dword_4F96DB8;
      result = sub_16AF4B0();
    }
    while ( v4 != 2 );
  }
  else
  {
    v1 = sub_22077B0(80);
    if ( v1 )
    {
      *(_QWORD *)(v1 + 8) = 23;
      *(_QWORD *)v1 = "External Alias Analysis";
      *(_QWORD *)(v1 + 16) = "external-aa";
      *(_QWORD *)(v1 + 32) = &unk_4F96DBC;
      *(_WORD *)(v1 + 40) = 256;
      *(_QWORD *)(v1 + 24) = 11;
      *(_BYTE *)(v1 + 42) = 0;
      *(_QWORD *)(v1 + 48) = 0;
      *(_QWORD *)(v1 + 56) = 0;
      *(_QWORD *)(v1 + 64) = 0;
      *(_QWORD *)(v1 + 72) = sub_134D7F0;
    }
    sub_163A800(a1, v1, 1);
    result = sub_16AF4B0();
    dword_4F96DB8 = 2;
  }
  return result;
}
