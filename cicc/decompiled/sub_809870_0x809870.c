// Function: sub_809870
// Address: 0x809870
//
__int64 __fastcall sub_809870(__int64 a1, const char *a2)
{
  __int64 v2; // rbx
  __int64 v3; // rax
  unsigned int v4; // r8d
  __int64 v5; // rdx
  const char *v7; // rdi
  __int64 v8; // rax
  _QWORD *v9; // r12
  _QWORD *v10; // rax
  __int64 v11; // rcx

  v2 = a1;
  v3 = sub_809820(a1);
  v4 = 0;
  if ( v3 )
  {
    v5 = *(_QWORD *)(v3 + 40);
    if ( v5 )
    {
      if ( *(_BYTE *)(v5 + 28) == 3 && (*(_BYTE *)(*(_QWORD *)(v5 + 32) + 124LL) & 0x10) != 0 )
      {
        v7 = *(const char **)(v3 + 8);
        if ( v7 )
        {
          v4 = strcmp(v7, a2);
          if ( v4 )
          {
            return 0;
          }
          else
          {
            while ( *(_BYTE *)(v2 + 140) == 12 )
              v2 = *(_QWORD *)(v2 + 160);
            v8 = *(_QWORD *)(*(_QWORD *)(v2 + 168) + 168LL);
            if ( v8 )
            {
              if ( !*(_BYTE *)(v8 + 8) && !*(_QWORD *)v8 )
              {
                v9 = *(_QWORD **)(v8 + 32);
                v10 = sub_72BA30(0);
                v4 = 1;
                if ( v9 != v10 )
                  return (unsigned int)sub_8D97D0(v9, v10, 0, v11, 1) != 0;
              }
            }
          }
        }
      }
    }
  }
  return v4;
}
