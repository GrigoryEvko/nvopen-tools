// Function: sub_34E0050
// Address: 0x34e0050
//
__int64 __fastcall sub_34E0050(_QWORD *a1, unsigned int *a2, _QWORD *a3)
{
  __int64 v5; // rax
  unsigned int v6; // esi
  _QWORD *v7; // rdx
  _QWORD *v8; // r8
  __int64 v9; // r12
  unsigned int v10; // ecx
  _QWORD *v11; // rax
  char v13; // di

  v5 = sub_22077B0(0x30u);
  v6 = *a2;
  v7 = (_QWORD *)a1[2];
  v8 = a1 + 1;
  v9 = v5;
  *(_DWORD *)(v5 + 32) = v6;
  *(_QWORD *)(v5 + 40) = *a3;
  if ( v7 )
  {
    while ( 1 )
    {
      v10 = *((_DWORD *)v7 + 8);
      v11 = (_QWORD *)v7[3];
      if ( v10 > v6 )
        v11 = (_QWORD *)v7[2];
      if ( !v11 )
        break;
      v7 = v11;
    }
    v13 = 1;
    if ( v8 != v7 )
      v13 = v10 > v6;
  }
  else
  {
    v7 = a1 + 1;
    v13 = 1;
  }
  sub_220F040(v13, v9, v7, a1 + 1);
  ++a1[5];
  return v9;
}
