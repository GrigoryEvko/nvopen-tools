// Function: sub_24DEAD0
// Address: 0x24dead0
//
__int64 __fastcall sub_24DEAD0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  __int64 v7; // rbx
  __int64 v8; // r13
  __int64 v9; // r12
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 *v12; // rcx
  __int64 v13; // rax
  __int64 *v14; // rax
  __int64 v15; // r14
  unsigned __int8 *v16; // rax
  __int64 v17; // rax
  unsigned __int8 *v18; // [rsp+0h] [rbp-40h]

  result = *a1;
  v7 = *(_QWORD *)(*a1 + 80);
  v8 = *a1 + 72;
  if ( v8 == v7 )
  {
    v9 = 0;
  }
  else
  {
    if ( !v7 )
      BUG();
    while ( 1 )
    {
      v9 = *(_QWORD *)(v7 + 32);
      result = v7 + 24;
      if ( v9 != v7 + 24 )
        break;
      v7 = *(_QWORD *)(v7 + 8);
      if ( v8 == v7 )
        return result;
      if ( !v7 )
        BUG();
    }
  }
  while ( v8 != v7 )
  {
    if ( !v9 )
      BUG();
    if ( *(_BYTE *)(v9 - 24) == 85 )
    {
      v11 = *(_QWORD *)(v9 - 56);
      if ( v11 )
      {
        if ( *(_BYTE *)v11
          || *(_QWORD *)(v11 + 24) != *(_QWORD *)(v9 + 56)
          || (*(_BYTE *)(v11 + 33) & 0x20) == 0
          || *(_DWORD *)(v11 + 36) != 48 )
        {
          goto LABEL_55;
        }
        v15 = v9 - 24;
        v16 = sub_BD3990(*(unsigned __int8 **)(v9 - 24 + 32 * (3LL - (*(_DWORD *)(v9 - 20) & 0x7FFFFFF))), a2);
        if ( *v16 == 3 && **((_BYTE **)v16 - 4) != 10 )
        {
          v18 = sub_BD3990(*(unsigned __int8 **)(v15 + 32 * (2LL - (*(_DWORD *)(v9 - 20) & 0x7FFFFFF))), a2);
          if ( v18 != (unsigned __int8 *)sub_B43CB0(v9 - 24) )
          {
            v17 = *((unsigned int *)a1 + 4);
            if ( v17 + 1 > (unsigned __int64)*((unsigned int *)a1 + 5) )
            {
              a2 = (__int64)(a1 + 3);
              sub_C8D5F0((__int64)(a1 + 1), a1 + 3, v17 + 1, 8u, a5, a6);
              v17 = *((unsigned int *)a1 + 4);
            }
            a3 = a1[1];
            *(_QWORD *)(a3 + 8 * v17) = v15;
            ++*((_DWORD *)a1 + 4);
          }
        }
        if ( *(_BYTE *)(v9 - 24) == 85 )
        {
          v11 = *(_QWORD *)(v9 - 56);
          if ( v11 )
          {
LABEL_55:
            if ( !*(_BYTE *)v11 )
            {
              v12 = *(__int64 **)(v9 + 56);
              if ( *(__int64 **)(v11 + 24) == v12 && (*(_BYTE *)(v11 + 33) & 0x20) != 0 && *(_DWORD *)(v11 + 36) == 60 )
              {
                v13 = *(_QWORD *)(v9 - 8);
                if ( v13 )
                {
                  if ( !*(_QWORD *)(v13 + 8) )
                  {
                    a2 = *(_QWORD *)(v13 + 24);
                    if ( *(_BYTE *)a2 == 32 && (*(_DWORD *)(a2 + 4) & 0x7FFFFFFu) >> 1 == 3 )
                    {
                      if ( !*((_BYTE *)a1 + 84) )
                        goto LABEL_49;
                      v14 = (__int64 *)a1[8];
                      a3 = *((unsigned int *)a1 + 19);
                      v12 = &v14[a3];
                      if ( v14 != v12 )
                      {
                        while ( a2 != *v14 )
                        {
                          if ( v12 == ++v14 )
                            goto LABEL_36;
                        }
                        goto LABEL_12;
                      }
LABEL_36:
                      if ( (unsigned int)a3 < *((_DWORD *)a1 + 18) )
                      {
                        *((_DWORD *)a1 + 19) = a3 + 1;
                        *v12 = a2;
                        ++a1[7];
                      }
                      else
                      {
LABEL_49:
                        sub_C8CC70((__int64)(a1 + 7), a2, a3, (__int64)v12, a5, a6);
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
LABEL_12:
    v9 = *(_QWORD *)(v9 + 8);
    a3 = 0;
    while ( 1 )
    {
      v10 = v7 - 24;
      if ( !v7 )
        v10 = 0;
      result = v10 + 48;
      if ( v9 != result )
        break;
      v7 = *(_QWORD *)(v7 + 8);
      if ( v8 == v7 )
        return result;
      if ( !v7 )
        BUG();
      v9 = *(_QWORD *)(v7 + 32);
    }
  }
  return result;
}
