// Function: sub_39A94A0
// Address: 0x39a94a0
//
void __fastcall sub_39A94A0(char *src, char *a2)
{
  char *v2; // r9
  __int64 v5; // r12
  char *v6; // rcx
  _DWORD *v7; // rsi
  char *v8; // rax
  signed __int64 v9; // rdx
  char *v10; // rdi
  char *v11; // rdx
  char *v12; // r15
  char *i; // rsi
  __int64 v14; // rdi
  _DWORD *v15; // rdx
  signed __int64 v16; // r8

  if ( src != a2 )
  {
    v2 = src + 8;
    if ( a2 != src + 8 )
    {
      do
      {
        v5 = *(_QWORD *)v2;
        v6 = *(char **)(*(_QWORD *)v2 + 104LL);
        v7 = *(_DWORD **)(*(_QWORD *)src + 96LL);
        v8 = *(char **)(*(_QWORD *)v2 + 96LL);
        v9 = *(_QWORD *)(*(_QWORD *)src + 104LL) - (_QWORD)v7;
        v10 = &v8[v9];
        if ( v6 - v8 <= v9 )
          v10 = *(char **)(*(_QWORD *)v2 + 104LL);
        if ( v8 == v10 )
        {
LABEL_15:
          if ( *(_DWORD **)(*(_QWORD *)src + 104LL) == v7 )
          {
LABEL_16:
            for ( i = v2; ; i -= 8 )
            {
              v14 = *((_QWORD *)i - 1);
              v15 = *(_DWORD **)(v14 + 96);
              v16 = *(_QWORD *)(v14 + 104) - (_QWORD)v15;
              if ( v6 - v8 > v16 )
                v6 = &v8[v16];
              if ( v6 == v8 )
              {
LABEL_25:
                if ( v15 == *(_DWORD **)(v14 + 104) )
                {
LABEL_26:
                  *(_QWORD *)i = v5;
                  v12 = v2 + 8;
                  goto LABEL_13;
                }
              }
              else
              {
                while ( *(_DWORD *)v8 >= *v15 )
                {
                  if ( *(_DWORD *)v8 > *v15 )
                    goto LABEL_26;
                  v8 += 4;
                  ++v15;
                  if ( v6 == v8 )
                    goto LABEL_25;
                }
              }
              *(_QWORD *)i = v14;
              v6 = *(char **)(v5 + 104);
              v8 = *(char **)(v5 + 96);
            }
          }
        }
        else
        {
          v11 = *(char **)(*(_QWORD *)v2 + 96LL);
          while ( *(_DWORD *)v11 >= *v7 )
          {
            if ( *(_DWORD *)v11 > *v7 )
              goto LABEL_16;
            v11 += 4;
            ++v7;
            if ( v10 == v11 )
              goto LABEL_15;
          }
        }
        v12 = v2 + 8;
        if ( src != v2 )
          memmove(src + 8, src, v2 - src);
        *(_QWORD *)src = v5;
LABEL_13:
        v2 = v12;
      }
      while ( a2 != v12 );
    }
  }
}
