// Function: sub_2914720
// Address: 0x2914720
//
void __fastcall sub_2914720(__int64 a1, __int64 a2, __int64 *a3, unsigned __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r14
  __int64 v7; // rsi
  __int64 *v8; // rax
  __int64 v9; // rax

  v6 = *(_QWORD *)(a2 + 16);
  if ( v6 )
  {
    while ( 1 )
    {
      v7 = *(_QWORD *)(v6 + 24);
      if ( *(_BYTE *)(a1 + 108) )
      {
        v8 = *(__int64 **)(a1 + 88);
        a4 = *(unsigned int *)(a1 + 100);
        a3 = &v8[a4];
        if ( v8 != a3 )
        {
          while ( v7 != *v8 )
          {
            if ( a3 == ++v8 )
              goto LABEL_14;
          }
          goto LABEL_7;
        }
LABEL_14:
        if ( (unsigned int)a4 < *(_DWORD *)(a1 + 96) )
        {
          *(_DWORD *)(a1 + 100) = a4 + 1;
          *a3 = v7;
          ++*(_QWORD *)(a1 + 80);
          goto LABEL_10;
        }
      }
      sub_C8CC70(a1 + 80, v7, (__int64)a3, a4, a5, a6);
      if ( (_BYTE)a3 )
      {
LABEL_10:
        v9 = *(unsigned int *)(a1 + 8);
        a4 = *(unsigned int *)(a1 + 12);
        if ( v9 + 1 > a4 )
        {
          sub_C8D5F0(a1, (const void *)(a1 + 16), v9 + 1, 8u, a5, a6);
          v9 = *(unsigned int *)(a1 + 8);
        }
        a3 = *(__int64 **)a1;
        *(_QWORD *)(*(_QWORD *)a1 + 8 * v9) = v6;
        ++*(_DWORD *)(a1 + 8);
        v6 = *(_QWORD *)(v6 + 8);
        if ( !v6 )
          return;
      }
      else
      {
LABEL_7:
        v6 = *(_QWORD *)(v6 + 8);
        if ( !v6 )
          return;
      }
    }
  }
}
