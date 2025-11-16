// Function: sub_3950920
// Address: 0x3950920
//
void __fastcall sub_3950920(__int64 *a1)
{
  __int64 i; // rbx
  _QWORD *v3; // rax
  __int64 v4; // rdi
  __int64 v5; // rdx
  __int64 v6; // r8
  __int64 v7; // rsi
  unsigned int v8; // ecx
  __int64 *v9; // rax
  __int64 v10; // r10
  __int64 v11; // r14
  __int64 v12; // r13
  unsigned int v13; // eax
  __int64 v14; // rax
  int v15; // eax
  int v16; // r11d

  for ( i = *a1; i; *a1 = i )
  {
    v3 = sub_1648700(i);
    if ( (unsigned __int8)(*((_BYTE *)v3 + 16) - 25) <= 9u )
    {
      v4 = a1[3];
      v5 = *(unsigned int *)(v4 + 48);
      if ( (_DWORD)v5 )
      {
        v6 = v3[5];
        v7 = *(_QWORD *)(v4 + 32);
        v8 = (v5 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
        v9 = (__int64 *)(v7 + 16LL * v8);
        v10 = *v9;
        if ( v6 == *v9 )
        {
LABEL_5:
          if ( v9 != (__int64 *)(v7 + 16 * v5) )
          {
            v11 = v9[1];
            if ( v11 )
            {
              v12 = a1[1];
              if ( v12 )
              {
                if ( v12 != v11 && v11 != *(_QWORD *)(v12 + 8) )
                {
                  if ( v12 == *(_QWORD *)(v11 + 8) || *(_DWORD *)(v11 + 16) >= *(_DWORD *)(v12 + 16) )
                    goto LABEL_14;
                  if ( *(_BYTE *)(v4 + 72) )
                  {
                    if ( *(_DWORD *)(v12 + 48) < *(_DWORD *)(v11 + 48) || *(_DWORD *)(v12 + 52) > *(_DWORD *)(v11 + 52) )
                      goto LABEL_14;
                  }
                  else
                  {
                    v13 = *(_DWORD *)(v4 + 76) + 1;
                    *(_DWORD *)(v4 + 76) = v13;
                    if ( v13 > 0x20 )
                    {
                      sub_15CC640(v4);
                      if ( *(_DWORD *)(v12 + 48) < *(_DWORD *)(v11 + 48)
                        || *(_DWORD *)(v12 + 52) > *(_DWORD *)(v11 + 52) )
                      {
                        goto LABEL_14;
                      }
                    }
                    else
                    {
                      do
                      {
                        v14 = v12;
                        v12 = *(_QWORD *)(v12 + 8);
                      }
                      while ( v12 && *(_DWORD *)(v11 + 16) <= *(_DWORD *)(v12 + 16) );
                      if ( v11 != v14 )
                      {
LABEL_14:
                        a1[2] = v11;
                        return;
                      }
                    }
                    i = *a1;
                  }
                }
              }
            }
          }
        }
        else
        {
          v15 = 1;
          while ( v10 != -8 )
          {
            v16 = v15 + 1;
            v8 = (v5 - 1) & (v15 + v8);
            v9 = (__int64 *)(v7 + 16LL * v8);
            v10 = *v9;
            if ( v6 == *v9 )
              goto LABEL_5;
            v15 = v16;
          }
        }
      }
    }
    i = *(_QWORD *)(i + 8);
  }
}
