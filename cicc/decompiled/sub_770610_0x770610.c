// Function: sub_770610
// Address: 0x770610
//
__int64 __fastcall sub_770610(unsigned __int64 a1, unsigned __int64 *a2)
{
  __int64 v2; // rax
  __int64 v3; // r9
  _QWORD *v4; // rbx
  unsigned __int64 v5; // rcx
  __int64 v6; // r8
  unsigned int i; // edx
  __int64 v8; // rax
  unsigned int v9; // r12d
  _QWORD *v10; // r15
  unsigned int j; // edx
  __int64 v12; // rax

  v2 = *(_QWORD *)(a1 + 112);
  v3 = *(_QWORD *)(v2 + 8);
  v4 = *(_QWORD **)(v2 + 16);
  v5 = *(_QWORD *)(v3 + 16);
  v6 = *(_QWORD *)(v5 + 40);
  for ( i = qword_4F08388 & (v5 >> 3); ; i = qword_4F08388 & (i + 1) )
  {
    v8 = qword_4F08380 + 16LL * i;
    if ( v5 == *(_QWORD *)v8 )
    {
      v9 = *(_DWORD *)(v8 + 8);
      goto LABEL_6;
    }
    if ( !*(_QWORD *)v8 )
      break;
  }
  v9 = 0;
LABEL_6:
  v10 = *(_QWORD **)v3;
  if ( *(_QWORD *)v3 != *v4 )
  {
    do
    {
      a1 = sub_8D5CF0(v6, *(_QWORD *)(v10[2] + 40LL));
      v6 = *(_QWORD *)(v10[2] + 40LL);
      for ( j = qword_4F08388 & (a1 >> 3); ; j = qword_4F08388 & (j + 1) )
      {
        v12 = qword_4F08380 + 16LL * j;
        if ( a1 == *(_QWORD *)v12 )
          break;
        if ( !*(_QWORD *)v12 )
          goto LABEL_12;
      }
      v9 += *(_DWORD *)(v12 + 8);
LABEL_12:
      v10 = (_QWORD *)*v10;
    }
    while ( (_QWORD *)*v4 != v10 );
  }
  if ( a2 )
    *a2 = a1;
  return v9;
}
