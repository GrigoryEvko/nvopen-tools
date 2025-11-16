// Function: sub_1869C50
// Address: 0x1869c50
//
__int64 __fastcall sub_1869C50(char a1, char a2, char a3)
{
  __int64 v4; // rax
  __int64 v5; // r12
  __int64 v6; // rax

  v4 = sub_22077B0(352);
  v5 = v4;
  if ( v4 )
  {
    sub_186A7A0(v4, &unk_4FAB3FC);
    *(_QWORD *)(v5 + 260) = 0;
    *(_WORD *)(v5 + 256) = 0;
    *(_QWORD *)v5 = off_49F1698;
    *(_QWORD *)(v5 + 268) = 0;
    sub_3851190(v5 + 276);
    v6 = sub_163A1D0();
    sub_1869A50(v6);
  }
  *(_BYTE *)(v5 + 257) = a1;
  *(_BYTE *)(v5 + 256) = a2;
  *(_BYTE *)(v5 + 153) = a3;
  return v5;
}
