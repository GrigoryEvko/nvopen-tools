// Function: sub_2D594F0
// Address: 0x2d594f0
//
void __fastcall sub_2D594F0(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rbx
  __int64 v8; // r12
  char i; // r13
  __int64 v10; // rsi
  __int64 *v11; // rax

  if ( *(_BYTE *)a1 > 0x1Cu )
  {
    v7 = *(_QWORD *)(a1 + 16);
    v8 = (__int64)a3;
    for ( i = a4; v7; v7 = *(_QWORD *)(v7 + 8) )
    {
      while ( !i )
      {
LABEL_4:
        v7 = *(_QWORD *)(v7 + 8);
        if ( !v7 )
          goto LABEL_12;
      }
      v10 = *(_QWORD *)(*(_QWORD *)(v7 + 24) + 40LL);
      if ( !*(_BYTE *)(v8 + 28) )
        goto LABEL_13;
      v11 = *(__int64 **)(v8 + 8);
      a4 = *(unsigned int *)(v8 + 20);
      a3 = &v11[a4];
      if ( v11 != a3 )
      {
        while ( v10 != *v11 )
        {
          if ( a3 == ++v11 )
            goto LABEL_10;
        }
        goto LABEL_4;
      }
LABEL_10:
      if ( (unsigned int)a4 >= *(_DWORD *)(v8 + 16) )
      {
LABEL_13:
        sub_C8CC70(v8, v10, (__int64)a3, a4, a5, a6);
        goto LABEL_4;
      }
      a4 = (unsigned int)(a4 + 1);
      *(_DWORD *)(v8 + 20) = a4;
      *a3 = v10;
      ++*(_QWORD *)v8;
    }
  }
LABEL_12:
  sub_BD84D0(a1, a2);
}
