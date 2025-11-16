// Function: sub_F58DD0
// Address: 0xf58dd0
//
void __fastcall sub_F58DD0(__int64 a1, __int64 *a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v6; // r13
  __int64 *v8; // rbx
  char v9; // di
  __int64 v10; // rsi
  __int64 *v11; // rax

  if ( a2 != a3 )
  {
    v6 = a3;
    v8 = a2;
    v9 = *(_BYTE *)(a1 + 28);
    do
    {
      v10 = *(_QWORD *)(v8[3] + 40);
      if ( !v9 )
        goto LABEL_12;
      v11 = *(__int64 **)(a1 + 8);
      a4 = *(unsigned int *)(a1 + 20);
      a3 = &v11[a4];
      if ( v11 != a3 )
      {
        while ( v10 != *v11 )
        {
          if ( a3 == ++v11 )
            goto LABEL_13;
        }
        goto LABEL_8;
      }
LABEL_13:
      if ( (unsigned int)a4 < *(_DWORD *)(a1 + 16) )
      {
        a4 = (unsigned int)(a4 + 1);
        *(_DWORD *)(a1 + 20) = a4;
        *a3 = v10;
        v9 = *(_BYTE *)(a1 + 28);
        ++*(_QWORD *)a1;
      }
      else
      {
LABEL_12:
        sub_C8CC70(a1, v10, (__int64)a3, a4, a5, a6);
        v9 = *(_BYTE *)(a1 + 28);
      }
      do
LABEL_8:
        v8 = (__int64 *)v8[1];
      while ( v8 && (unsigned __int8)(*(_BYTE *)v8[3] - 30) > 0xAu );
    }
    while ( v6 != v8 );
  }
}
