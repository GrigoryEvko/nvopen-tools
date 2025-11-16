// Function: sub_1DD9280
// Address: 0x1dd9280
//
void __fastcall sub_1DD9280(__int64 a1, _QWORD *a2)
{
  __int64 *i; // rax
  __int64 v5; // r12
  int *v6; // rax
  __int64 v7; // rsi
  __int64 v8; // r12
  __int64 v9; // rsi
  int v10; // eax
  int v11; // edi
  unsigned int v12; // eax
  __int64 v13; // rdx

  if ( (_QWORD *)a1 != a2 )
  {
    for ( i = (__int64 *)a2[11]; i != (__int64 *)a2[12]; i = (__int64 *)a2[11] )
    {
LABEL_6:
      v5 = *i;
      v6 = (int *)a2[14];
      if ( v6 == (int *)a2[15] )
        sub_1DD8D40(a1, v5);
      else
        sub_1DD8FE0(a1, v5, *v6);
      v7 = v5;
      v8 = v5 + 24;
      sub_1DD91B0((__int64)a2, v7);
      v9 = *(_QWORD *)(v8 + 8);
      if ( v9 != v8 )
      {
        while ( **(_WORD **)(v9 + 16) == 45 || !**(_WORD **)(v9 + 16) )
        {
          v10 = *(_DWORD *)(v9 + 40);
          v11 = v10 + 1;
          if ( v10 != 1 )
          {
            v12 = 2;
            do
            {
              while ( 1 )
              {
                v13 = *(_QWORD *)(v9 + 32) + 40LL * v12;
                if ( a2 == *(_QWORD **)(v13 + 24) )
                  break;
                v12 += 2;
                if ( v11 == v12 )
                  goto LABEL_15;
              }
              v12 += 2;
              *(_QWORD *)(v13 + 24) = a1;
            }
            while ( v11 != v12 );
          }
LABEL_15:
          v9 = *(_QWORD *)(v9 + 8);
          if ( v8 == v9 )
          {
            i = (__int64 *)a2[11];
            if ( i != (__int64 *)a2[12] )
              goto LABEL_6;
            goto LABEL_17;
          }
        }
      }
    }
LABEL_17:
    sub_1D96570(*(unsigned int **)(a1 + 112), *(unsigned int **)(a1 + 120));
  }
}
