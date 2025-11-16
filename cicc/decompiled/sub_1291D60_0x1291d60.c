// Function: sub_1291D60
// Address: 0x1291d60
//
void __fastcall sub_1291D60(__int64 a1, __int64 a2)
{
  __int64 *v2; // rbx
  __int64 v4; // r13
  unsigned __int64 v5; // rsi
  __int64 v6; // rax
  __int64 v7; // rdi
  unsigned int v8; // ecx
  __int64 *v9; // rdx
  __int64 v10; // r9
  bool v11; // zf
  __int64 v12; // rax
  _BYTE *v13; // rsi
  int v14; // edx
  int v15; // r10d
  _QWORD v16[6]; // [rsp-30h] [rbp-30h] BYREF

  if ( *(_BYTE *)(a1 + 168) )
  {
    v2 = *(__int64 **)(a2 + 72);
    if ( v2 )
    {
      v4 = a1 + 176;
      do
      {
        while ( 1 )
        {
          if ( *((_BYTE *)v2 + 8) == 7 )
          {
            v5 = v2[2];
            v16[0] = v5;
            if ( (*(_BYTE *)(v5 + 170) & 0x60) == 0 && *(_BYTE *)(v5 + 177) != 5 )
            {
              v6 = *(unsigned int *)(a1 + 24);
              if ( (_DWORD)v6 )
              {
                v7 = *(_QWORD *)(a1 + 8);
                v8 = (v6 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
                v9 = (__int64 *)(v7 + 16LL * v8);
                v10 = *v9;
                if ( v5 != *v9 )
                {
                  v14 = 1;
                  while ( v10 != -8 )
                  {
                    v15 = v14 + 1;
                    v8 = (v6 - 1) & (v14 + v8);
                    v9 = (__int64 *)(v7 + 16LL * v8);
                    v10 = *v9;
                    if ( v5 == *v9 )
                      goto LABEL_10;
                    v14 = v15;
                  }
                  goto LABEL_4;
                }
LABEL_10:
                if ( v9 != (__int64 *)(v7 + 16 * v6) )
                  break;
              }
            }
          }
LABEL_4:
          v2 = (__int64 *)*v2;
          if ( !v2 )
            return;
        }
        v11 = !sub_127C9A0(v4, v5);
        v12 = *(_QWORD *)(a1 + 352);
        v13 = *(_BYTE **)(v12 - 16);
        if ( !v11 )
        {
          if ( v13 == *(_BYTE **)(v12 - 8) )
          {
            sub_930F50(v12 - 24, v13, v16);
          }
          else
          {
            if ( v13 )
            {
              *(_QWORD *)v13 = v16[0];
              v13 = *(_BYTE **)(v12 - 16);
            }
            *(_QWORD *)(v12 - 16) = v13 + 8;
          }
          goto LABEL_4;
        }
        if ( v13 == *(_BYTE **)(v12 - 8) )
        {
          sub_930F50(v12 - 24, v13, v16);
        }
        else
        {
          if ( v13 )
          {
            *(_QWORD *)v13 = v16[0];
            v13 = *(_BYTE **)(v12 - 16);
          }
          *(_QWORD *)(v12 - 16) = v13 + 8;
        }
        sub_12A5710(a1, v16[0], 0);
        v2 = (__int64 *)*v2;
      }
      while ( v2 );
    }
  }
}
