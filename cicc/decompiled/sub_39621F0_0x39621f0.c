// Function: sub_39621F0
// Address: 0x39621f0
//
void __fastcall sub_39621F0(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rcx
  __int64 v3; // r8
  __int64 v4; // r9
  int v5; // r12d

  if ( (unsigned int)sub_16AF4C0(&dword_5054CC8, 1, 0) )
  {
    do
    {
      v5 = dword_5054CC8;
      sub_16AF4B0();
    }
    while ( v5 != 2 );
  }
  else
  {
    sub_13FBE20(a1);
    sub_15CD350(a1);
    sub_1368E50(a1);
    sub_14A3D70(a1);
    sub_1BFB430(a1);
    v1 = sub_22077B0(0x50u);
    if ( v1 )
    {
      *(_QWORD *)(v1 + 8) = 37;
      *(_QWORD *)v1 = "Register Rematerialization on NVVM IR";
      *(_QWORD *)(v1 + 16) = "remat";
      *(_QWORD *)(v1 + 24) = 5;
      *(_QWORD *)(v1 + 32) = &unk_5054CCC;
      *(_WORD *)(v1 + 40) = 0;
      *(_BYTE *)(v1 + 42) = 0;
      *(_QWORD *)(v1 + 48) = 0;
      *(_QWORD *)(v1 + 56) = 0;
      *(_QWORD *)(v1 + 64) = 0;
      *(_QWORD *)(v1 + 72) = sub_39622F0;
    }
    sub_163A800(a1, (_QWORD *)v1, 1, v2, v3, v4);
    sub_16AF4B0();
    dword_5054CC8 = 2;
  }
}
