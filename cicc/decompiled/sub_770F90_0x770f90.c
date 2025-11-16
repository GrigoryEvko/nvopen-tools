// Function: sub_770F90
// Address: 0x770f90
//
void __fastcall sub_770F90(__int64 *a1, __int64 a2, __int64 a3)
{
  _QWORD *v4; // r12
  _QWORD *v5; // r15
  __int64 i; // rbx
  unsigned __int64 v8; // rcx
  unsigned int j; // edx
  __int64 v10; // rax

  v4 = *(_QWORD **)(a3 + 16);
  v5 = *(_QWORD **)(a3 + 8);
  for ( i = *a1; (_QWORD *)*v4 != v5; v5 = (_QWORD *)*v5 )
  {
    v8 = sub_8D5CF0(a2, *(_QWORD *)(v5[2] + 40LL));
    for ( j = qword_4F08388 & (v8 >> 3); ; j = qword_4F08388 & (j + 1) )
    {
      v10 = qword_4F08380 + 16LL * j;
      if ( v8 == *(_QWORD *)v10 )
        break;
      if ( !*(_QWORD *)v10 )
        goto LABEL_7;
    }
    i += *(unsigned int *)(v10 + 8);
LABEL_7:
    a2 = *(_QWORD *)(v8 + 40);
  }
  *a1 = i;
  *((_BYTE *)a1 + 8) &= ~8u;
}
