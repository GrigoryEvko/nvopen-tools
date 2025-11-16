// Function: sub_71B6A0
// Address: 0x71b6a0
//
__int64 __fastcall sub_71B6A0(__int64 a1, __int64 a2, __int64 a3, _DWORD *a4)
{
  int v6; // edx
  int v7; // edx
  __int64 v8; // r12
  __int64 v9; // rax
  bool v10; // zf

  *a4 = sub_8D0B70(*(_QWORD *)a1);
  v6 = unk_4F04C2C;
  unk_4F04C2C = -1;
  a4[1] = v6;
  v7 = dword_4F04C44;
  dword_4F04C44 = -1;
  a4[2] = v7;
  sub_865D70(a3, 1, 0, 1, 1, 0);
  v8 = sub_8600D0(17, 0xFFFFFFFFLL, 0, a1);
  if ( *(_BYTE *)(a1 + 172) == 1 )
    *(_BYTE *)(a1 + 172) = 0;
  v9 = *(_QWORD *)(a2 + 168);
  v10 = *(_QWORD *)(v9 + 40) == 0;
  *(_QWORD *)(v9 + 8) = a1;
  if ( !v10 )
    *(_QWORD *)(v8 + 64) = sub_71B620(*(_QWORD *)(a1 + 152));
  return v8;
}
