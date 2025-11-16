// Function: sub_5C9200
// Address: 0x5c9200
//
__int64 __fastcall sub_5C9200(__int64 a1, __int64 a2)
{
  __int64 *v3; // r14
  const char *v5; // rbx
  size_t v6; // rax
  _BYTE *v7; // rdi
  unsigned __int64 v8; // r15
  _BYTE *v9; // rbx
  int v10; // esi
  _BYTE *v11; // rax
  int v12; // ecx
  _DWORD v14[13]; // [rsp+1Ch] [rbp-34h] BYREF

  v3 = *(__int64 **)(a1 + 32);
  if ( *((_BYTE *)v3 + 10) == 1 )
  {
    if ( *(_BYTE *)v3[5] == 34 )
    {
      if ( unk_4F077C4 != 2 )
      {
        v14[0] = 0;
        do
        {
          v5 = (const char *)v3[5];
          v6 = strlen(v5);
          if ( *v5 == 44 )
          {
            if ( v6 != 1 )
              goto LABEL_28;
          }
          else
          {
            if ( v6 <= 1 || *v5 != 34 )
            {
LABEL_28:
              sub_6851C0(1038, v3 + 3);
              *(_BYTE *)(a1 + 8) = 0;
              return a2;
            }
            v7 = v5 + 1;
            v8 = (unsigned __int64)&v5[v6 - 1];
            while ( (unsigned __int64)v7 < v8 )
            {
              v9 = v7;
              v10 = 0;
              while ( *v9 != 44 )
              {
                ++v9;
                ++v10;
                if ( v9 == (_BYTE *)v8 )
                  goto LABEL_17;
              }
              if ( !v10 )
                continue;
LABEL_17:
              if ( unk_4F077B4 && *v7 == 32 )
              {
                v11 = v7;
                do
                  v12 = v10 + (_DWORD)v7 - (_DWORD)++v11;
                while ( *v11 == 32 );
                v7 = v11;
                v10 = v12;
              }
              sub_88B6E0(v7, v10, v3, a2, v14);
              v7 = &v9[*v9 == 44];
            }
            if ( v14[0] )
            {
              *(_BYTE *)(a1 + 8) = 0;
              return a2;
            }
          }
          v3 = (__int64 *)*v3;
        }
        while ( v3 && *((_BYTE *)v3 + 10) == 1 );
      }
    }
    else
    {
      sub_6851C0(1038, v3 + 3);
      *(_BYTE *)(a1 + 8) = 0;
    }
  }
  return a2;
}
