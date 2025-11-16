// Function: sub_31C6E20
// Address: 0x31c6e20
//
void __fastcall sub_31C6E20(_QWORD *a1, unsigned __int64 a2)
{
  _QWORD *v2; // r12
  _QWORD *v3; // rax
  unsigned __int64 v4; // rbx
  _QWORD *v5; // rsi
  __int64 v6; // rcx
  _QWORD *v7; // rdx
  _QWORD *v8; // rax
  _QWORD *v9; // r14
  char v10; // bl
  __int64 v11; // rax
  _BYTE *v12; // rsi
  __int64 v13; // r12
  __int64 v14; // r14
  _BYTE *v15; // rsi
  unsigned __int64 v16[7]; // [rsp+8h] [rbp-38h] BYREF

  v2 = a1 + 27;
  v3 = (_QWORD *)a1[28];
  v16[0] = a2;
  if ( !v3 )
    goto LABEL_8;
  v4 = a2;
  v5 = a1 + 27;
  do
  {
    while ( 1 )
    {
      v6 = v3[2];
      v7 = (_QWORD *)v3[3];
      if ( v3[4] >= v4 )
        break;
      v3 = (_QWORD *)v3[3];
      if ( !v7 )
        goto LABEL_6;
    }
    v5 = v3;
    v3 = (_QWORD *)v3[2];
  }
  while ( v6 );
LABEL_6:
  if ( v2 == v5 || v5[4] > v4 )
  {
LABEL_8:
    v8 = sub_2D11AF0((__int64)(a1 + 26), v16);
    v9 = v7;
    if ( v7 )
    {
      v10 = v8 || v2 == v7 || v16[0] < v7[4];
      v11 = sub_22077B0(0x28u);
      *(_QWORD *)(v11 + 32) = v16[0];
      sub_220F040(v10, v11, v9, v2);
      ++a1[31];
    }
    v12 = (_BYTE *)a1[24];
    if ( v12 == (_BYTE *)a1[25] )
    {
      sub_24454E0((__int64)(a1 + 23), v12, v16);
      v4 = v16[0];
    }
    else
    {
      v4 = v16[0];
      if ( v12 )
      {
        *(_QWORD *)v12 = v16[0];
        v12 = (_BYTE *)a1[24];
        v4 = v16[0];
      }
      a1[24] = v12 + 8;
    }
  }
  if ( *(_BYTE *)v4 == 84 && (*(_DWORD *)(v4 + 4) & 0x7FFFFFF) != 0 )
  {
    v13 = 0;
    v14 = 32LL * (*(_DWORD *)(v4 + 4) & 0x7FFFFFF);
    do
    {
      v15 = *(_BYTE **)(*(_QWORD *)(v4 - 8) + v13);
      if ( *v15 > 0x1Cu )
        sub_31C6E20(a1, v15, v7, v6);
      v13 += 32;
    }
    while ( v14 != v13 );
  }
}
