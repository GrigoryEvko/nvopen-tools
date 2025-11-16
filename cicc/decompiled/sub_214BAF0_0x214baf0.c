// Function: sub_214BAF0
// Address: 0x214baf0
//
__int64 __fastcall sub_214BAF0(__int64 a1, __int64 a2)
{
  __int64 i; // rbx
  _QWORD *v4; // rdi
  unsigned __int8 v5; // al
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rdi
  unsigned int v10; // esi
  __int64 *v11; // rcx
  __int64 v12; // r9
  int v14; // ecx
  int v15; // r10d

  for ( i = *(_QWORD *)(a1 + 8); i; i = *(_QWORD *)(i + 8) )
  {
    v4 = sub_1648700(i);
    v5 = *((_BYTE *)v4 + 16);
    if ( v5 <= 0x10u )
    {
      if ( (unsigned __int8)sub_214BAF0(v4, a2) )
        return 1;
    }
    else if ( v5 > 0x17u )
    {
      v6 = v4[5];
      if ( v6 )
      {
        v7 = *(_QWORD *)(v6 + 56);
        if ( v7 )
        {
          v8 = *(unsigned int *)(a2 + 24);
          if ( (_DWORD)v8 )
          {
            v9 = *(_QWORD *)(a2 + 8);
            v10 = (v8 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
            v11 = (__int64 *)(v9 + 16LL * v10);
            v12 = *v11;
            if ( v7 == *v11 )
            {
LABEL_11:
              if ( v11 != (__int64 *)(v9 + 16 * v8) )
                return 1;
            }
            else
            {
              v14 = 1;
              while ( v12 != -8 )
              {
                v15 = v14 + 1;
                v10 = (v8 - 1) & (v14 + v10);
                v11 = (__int64 *)(v9 + 16LL * v10);
                v12 = *v11;
                if ( v7 == *v11 )
                  goto LABEL_11;
                v14 = v15;
              }
            }
          }
        }
      }
    }
  }
  return 0;
}
