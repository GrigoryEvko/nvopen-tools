// Function: sub_1D46400
// Address: 0x1d46400
//
void __fastcall sub_1D46400(_QWORD *a1, __int64 a2, __int64 a3)
{
  _QWORD *v5; // rax
  __int64 v6; // rdx
  _QWORD *v7; // rax
  __int64 i; // rdx
  __int64 *v9; // rax
  __int64 v10; // rcx
  __int64 v11; // r8
  _QWORD *v12; // rax
  _QWORD *j; // rdx

  if ( a3 && *(__int16 *)(a3 + 24) >= 0 )
  {
    v5 = (_QWORD *)a1[3];
    if ( a2 == *v5 )
      *v5 = a3;
    v6 = a1[4];
    v7 = *(_QWORD **)v6;
    for ( i = *(_QWORD *)v6 + 24LL * *(unsigned int *)(v6 + 8); (_QWORD *)i != v7; v7 += 3 )
    {
      while ( a2 != *v7 )
      {
        v7 += 3;
        if ( (_QWORD *)i == v7 )
          goto LABEL_10;
      }
      *v7 = a3;
    }
LABEL_10:
    v9 = (__int64 *)a1[5];
    v10 = *v9;
    v11 = *v9 + 136LL * *((unsigned int *)v9 + 2);
    if ( *v9 != v11 )
    {
      do
      {
        v12 = *(_QWORD **)(v10 + 8);
        for ( j = &v12[2 * *(unsigned int *)(v10 + 16)]; v12 != j; v12 += 2 )
        {
          while ( a2 != *v12 )
          {
            v12 += 2;
            if ( v12 == j )
              goto LABEL_16;
          }
          *v12 = a3;
        }
LABEL_16:
        v10 += 136;
      }
      while ( v11 != v10 );
    }
  }
}
