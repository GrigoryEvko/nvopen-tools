// Function: sub_24B1CF0
// Address: 0x24b1cf0
//
void __fastcall sub_24B1CF0(_QWORD *a1, unsigned __int64 *a2)
{
  _QWORD *i; // rbx
  unsigned __int64 v3; // r15
  _QWORD *v4; // rax
  _QWORD *j; // rbx
  unsigned __int64 v6; // r15
  _QWORD *v7; // rax
  _QWORD *k; // rbx
  __int64 v9; // r15
  unsigned __int64 v10; // r12
  _QWORD *v11; // rax

  for ( i = (_QWORD *)a1[4]; a1 + 3 != i; i = (_QWORD *)i[1] )
  {
    if ( !i )
      BUG();
    v3 = *(i - 1);
    if ( v3 )
    {
      v4 = (_QWORD *)sub_22077B0(0x18u);
      if ( v4 )
        *v4 = 0;
      v4[1] = v3;
      v4[2] = i - 7;
      sub_24B19A0(a2, 0, v4 + 1, v3, (unsigned __int64)v4);
    }
  }
  for ( j = (_QWORD *)a1[2]; a1 + 1 != j; j = (_QWORD *)j[1] )
  {
    if ( !j )
      BUG();
    v6 = *(j - 1);
    if ( v6 )
    {
      v7 = (_QWORD *)sub_22077B0(0x18u);
      if ( v7 )
        *v7 = 0;
      v7[1] = v6;
      v7[2] = j - 7;
      sub_24B19A0(a2, 0, v7 + 1, v6, (unsigned __int64)v7);
    }
  }
  for ( k = (_QWORD *)a1[6]; a1 + 5 != k; k = (_QWORD *)k[1] )
  {
    v9 = 0;
    if ( k )
      v9 = (__int64)(k - 6);
    v10 = sub_B326A0(v9);
    if ( v10 )
    {
      v11 = (_QWORD *)sub_22077B0(0x18u);
      if ( v11 )
        *v11 = 0;
      v11[1] = v10;
      v11[2] = v9;
      sub_24B19A0(a2, 0, v11 + 1, v10, (unsigned __int64)v11);
    }
  }
}
