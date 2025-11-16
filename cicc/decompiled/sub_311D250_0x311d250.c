// Function: sub_311D250
// Address: 0x311d250
//
void __fastcall sub_311D250(int *a1, int *a2)
{
  unsigned int *v2; // r9
  int *v3; // r11
  int *i; // r8
  unsigned int v5; // ecx
  unsigned int v6; // eax
  __int64 v7; // rdi
  int *v8; // r10
  unsigned int v9; // esi
  unsigned __int64 v10; // rbx
  __int64 v11; // rax
  int v12; // edx

  if ( a1 != a2 )
  {
    v2 = (unsigned int *)a1;
    v3 = a2;
    if ( a2 != a1 + 4 )
    {
      for ( i = a1 + 8; ; i += 4 )
      {
        v5 = *(i - 4);
        v6 = *v2;
        v7 = (__int64)(i - 4);
        v8 = i;
        if ( v5 < *v2 )
          break;
        if ( v5 == v6 )
        {
          v9 = *(i - 3);
          if ( v9 < v2[1] )
          {
            v10 = *((_QWORD *)i - 1);
            goto LABEL_6;
          }
        }
        if ( v5 <= v6 )
        {
          v9 = *(i - 3);
          if ( v2[1] >= v9 )
          {
            v10 = *((_QWORD *)i - 1);
            if ( v10 < *((_QWORD *)v2 + 1) )
              goto LABEL_6;
          }
        }
        sub_311D1F0(v7);
LABEL_9:
        if ( v3 == v8 )
          return;
      }
      v9 = *(i - 3);
      v10 = *((_QWORD *)i - 1);
LABEL_6:
      v11 = (v7 - (__int64)v2) >> 4;
      if ( v7 - (__int64)v2 > 0 )
      {
        do
        {
          v12 = *(_DWORD *)(v7 - 16);
          v7 -= 16;
          *(_DWORD *)(v7 + 16) = v12;
          *(_DWORD *)(v7 + 20) = *(_DWORD *)(v7 + 4);
          *(_QWORD *)(v7 + 24) = *(_QWORD *)(v7 + 8);
          --v11;
        }
        while ( v11 );
      }
      *v2 = v5;
      v2[1] = v9;
      *((_QWORD *)v2 + 1) = v10;
      goto LABEL_9;
    }
  }
}
