// Function: sub_31200F0
// Address: 0x31200f0
//
void __fastcall sub_31200F0(__int64 a1, unsigned int *a2, unsigned __int64 a3)
{
  __int64 v3; // r13
  unsigned int *v5; // rbx
  __int64 i; // r15
  __int64 v7; // r13
  unsigned __int64 v8; // rcx
  unsigned __int64 v9; // r8
  unsigned int v10; // edx
  unsigned int v11; // eax

  v3 = (__int64)a2 - a1;
  v5 = a2;
  if ( (__int64)a2 - a1 > 16 )
  {
    for ( i = ((v3 >> 4) - 2) / 2; ; --i )
    {
      sub_311D5D0(a1, i, v3 >> 4, *(_QWORD *)(a1 + 16 * i), *(_QWORD *)(a1 + 16 * i + 8));
      if ( !i )
        break;
    }
  }
  v7 = v3 >> 4;
  if ( (unsigned __int64)a2 < a3 )
  {
    while ( 1 )
    {
      v10 = *v5;
      v11 = *(_DWORD *)a1;
      if ( *v5 < *(_DWORD *)a1 || v10 == v11 && v5[1] < *(_DWORD *)(a1 + 4) )
        goto LABEL_7;
      if ( v10 > v11 || *(_DWORD *)(a1 + 4) < v5[1] )
        goto LABEL_8;
      if ( *((_QWORD *)v5 + 1) < *(_QWORD *)(a1 + 8) )
      {
LABEL_7:
        v8 = *(_QWORD *)v5;
        *v5 = v11;
        v9 = *((_QWORD *)v5 + 1);
        v5[1] = *(_DWORD *)(a1 + 4);
        *((_QWORD *)v5 + 1) = *(_QWORD *)(a1 + 8);
        sub_311D5D0(a1, 0, v7, v8, v9);
LABEL_8:
        v5 += 4;
        if ( a3 <= (unsigned __int64)v5 )
          return;
      }
      else
      {
        v5 += 4;
        if ( a3 <= (unsigned __int64)v5 )
          return;
      }
    }
  }
}
