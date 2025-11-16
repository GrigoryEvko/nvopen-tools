// Function: sub_28392C0
// Address: 0x28392c0
//
__int64 __fastcall sub_28392C0(_BYTE **a1, _BYTE **a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  _BYTE **v7; // rbx
  __int64 v9; // rsi
  __int64 *v10; // rax

  v6 = (__int64)a3;
  if ( a1 != a2 )
  {
    v7 = a1;
    while ( 1 )
    {
      while ( **v7 != 62 )
      {
LABEL_9:
        if ( a2 == ++v7 )
          return v6;
      }
      v9 = *((_QWORD *)*v7 - 4);
      if ( *(_BYTE *)(v6 + 28) )
      {
        v10 = *(__int64 **)(v6 + 8);
        a4 = *(unsigned int *)(v6 + 20);
        a3 = &v10[a4];
        if ( v10 != a3 )
        {
          while ( v9 != *v10 )
          {
            if ( a3 == ++v10 )
              goto LABEL_13;
          }
          goto LABEL_9;
        }
LABEL_13:
        if ( (unsigned int)a4 >= *(_DWORD *)(v6 + 16) )
          goto LABEL_11;
        a4 = (unsigned int)(a4 + 1);
        ++v7;
        *(_DWORD *)(v6 + 20) = a4;
        *a3 = v9;
        ++*(_QWORD *)v6;
        if ( a2 == v7 )
          return v6;
      }
      else
      {
LABEL_11:
        ++v7;
        sub_C8CC70(v6, v9, (__int64)a3, a4, a5, a6);
        if ( a2 == v7 )
          return v6;
      }
    }
  }
  return v6;
}
