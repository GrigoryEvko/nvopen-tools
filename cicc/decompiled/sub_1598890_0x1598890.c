// Function: sub_1598890
// Address: 0x1598890
//
__int64 __fastcall sub_1598890(__int64 a1, __int64 a2, _QWORD *a3)
{
  __int64 result; // rax
  int v5; // ecx
  __int64 v6; // r10
  unsigned int v7; // r9d
  _QWORD *v8; // rdi
  __int64 v9; // rdx
  _QWORD *v10; // r13
  int i; // r12d
  unsigned int v12; // r11d
  _QWORD *v13; // rax
  _QWORD *v14; // rdx
  __int64 v15; // r11

  result = *(unsigned int *)(a1 + 24);
  if ( (_DWORD)result )
  {
    v5 = result - 1;
    v6 = *(_QWORD *)(a1 + 8);
    v7 = (result - 1) & *(_DWORD *)a2;
    v8 = (_QWORD *)(v6 + 8LL * v7);
    v9 = *v8;
    if ( *v8 == -8 )
    {
      *a3 = v8;
      return 0;
    }
    else
    {
      v10 = 0;
      for ( i = 1; ; ++i )
      {
        if ( v9 == -16 )
        {
          if ( !v10 )
            v10 = v8;
        }
        else if ( *(_QWORD *)(a2 + 8) == *(_QWORD *)v9 )
        {
          v12 = *(_DWORD *)(v9 + 20) & 0xFFFFFFF;
          if ( *(_QWORD *)(a2 + 24) == v12 )
          {
            if ( !v12 )
            {
LABEL_14:
              *a3 = v8;
              return 1;
            }
            v13 = *(_QWORD **)(a2 + 16);
            v14 = (_QWORD *)(v9 - 24LL * v12);
            v15 = (__int64)&v13[v12];
            while ( *v13 == *v14 )
            {
              ++v13;
              v14 += 3;
              if ( v13 == (_QWORD *)v15 )
                goto LABEL_14;
            }
          }
        }
        v7 = v5 & (i + v7);
        v8 = (_QWORD *)(v6 + 8LL * v7);
        v9 = *v8;
        if ( *v8 == -8 )
          break;
      }
      if ( v10 )
        v8 = v10;
      *a3 = v8;
      return 0;
    }
  }
  else
  {
    *a3 = 0;
  }
  return result;
}
