// Function: sub_13A5DC0
// Address: 0x13a5dc0
//
__int64 __fastcall sub_13A5DC0(__int64 a1)
{
  __int64 v1; // rax
  __int64 result; // rax
  int v3; // r12d

  if ( (unsigned int)sub_16AF4C0(&dword_4F98D30, 1, 0) )
  {
    do
    {
      v3 = dword_4F98D30;
      result = sub_16AF4B0();
    }
    while ( v3 != 2 );
  }
  else
  {
    sub_13FBE20(a1);
    sub_1458320(a1);
    sub_134D8E0(a1);
    v1 = sub_22077B0(80);
    if ( v1 )
    {
      *(_QWORD *)(v1 + 8) = 19;
      *(_QWORD *)v1 = "Dependence Analysis";
      *(_QWORD *)(v1 + 16) = "da";
      *(_QWORD *)(v1 + 24) = 2;
      *(_QWORD *)(v1 + 32) = &unk_4F98D2D;
      *(_WORD *)(v1 + 40) = 257;
      *(_BYTE *)(v1 + 42) = 0;
      *(_QWORD *)(v1 + 48) = 0;
      *(_QWORD *)(v1 + 56) = 0;
      *(_QWORD *)(v1 + 64) = 0;
      *(_QWORD *)(v1 + 72) = sub_13A5EB0;
    }
    sub_163A800(a1, v1, 1);
    result = sub_16AF4B0();
    dword_4F98D30 = 2;
  }
  return result;
}
