// Function: sub_185C560
// Address: 0x185c560
//
__int64 __fastcall sub_185C560(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 i; // r15
  _QWORD *v7; // r12
  char v8; // al
  unsigned int v9; // ecx
  __int64 v10; // rax
  __int64 v11; // rax
  _QWORD *v13; // rax
  char v14; // dl
  char v15; // al
  _QWORD *v16; // rdi
  unsigned int v17; // r8d
  _QWORD *v18; // rsi
  unsigned __int8 v19; // [rsp+Fh] [rbp-31h]

  for ( i = *(_QWORD *)(a1 + 8); i; i = *(_QWORD *)(i + 8) )
  {
    v7 = sub_1648700(i);
    v8 = *((_BYTE *)v7 + 16);
    LOBYTE(v9) = v8 == 54 || (unsigned __int8)(v8 - 75) <= 1u;
    if ( !(_BYTE)v9 )
    {
      if ( v8 == 55 )
      {
        v10 = *(v7 - 6);
        if ( v10 )
        {
          if ( a1 == v10 )
          {
            v11 = *(v7 - 3);
            if ( a2 != v11 || !v11 )
              return v9;
          }
        }
      }
      else
      {
        switch ( v8 )
        {
          case '8':
            if ( (*((_DWORD *)v7 + 5) & 0xFFFFFFFu) <= 2 )
              return v9;
            break;
          case 'M':
            v13 = *(_QWORD **)(a3 + 8);
            if ( *(_QWORD **)(a3 + 16) != v13 )
              goto LABEL_14;
            v16 = &v13[*(unsigned int *)(a3 + 28)];
            v17 = *(_DWORD *)(a3 + 28);
            if ( v13 != v16 )
            {
              v18 = 0;
              while ( v7 != (_QWORD *)*v13 )
              {
                if ( *v13 == -2 )
                  v18 = v13;
                if ( v16 == ++v13 )
                {
                  if ( !v18 )
                    goto LABEL_30;
                  *v18 = v7;
                  --*(_DWORD *)(a3 + 32);
                  ++*(_QWORD *)a3;
                  goto LABEL_15;
                }
              }
              continue;
            }
LABEL_30:
            if ( v17 < *(_DWORD *)(a3 + 24) )
            {
              *(_DWORD *)(a3 + 28) = v17 + 1;
              *v16 = v7;
              ++*(_QWORD *)a3;
            }
            else
            {
LABEL_14:
              sub_16CCBA0(a3, (__int64)v7);
              LOBYTE(v9) = 0;
              if ( !v14 )
                continue;
            }
            break;
          case 'G':
            break;
          default:
            return v9;
        }
LABEL_15:
        v19 = v9;
        v15 = sub_185C560(v7, a2, a3);
        v9 = v19;
        if ( !v15 )
          return v9;
      }
    }
  }
  return 1;
}
