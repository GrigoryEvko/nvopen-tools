// Function: sub_D37540
// Address: 0xd37540
//
void __fastcall sub_D37540(__int64 a1)
{
  _QWORD *v2; // rax
  _QWORD *v3; // r12
  _QWORD *v4; // rbx
  _QWORD *v5; // rax
  __int64 v6; // rax
  int v7; // eax
  __int64 v8; // rcx
  int v9; // esi
  unsigned int v10; // eax
  __int64 *v11; // r13
  __int64 v12; // rdi
  __int64 *v13; // rdi
  int v14; // r8d

  if ( *(_DWORD *)(a1 + 16) )
  {
    v2 = *(_QWORD **)(a1 + 8);
    v3 = &v2[2 * *(unsigned int *)(a1 + 24)];
    if ( v2 != v3 )
    {
      while ( 1 )
      {
        v4 = v2;
        if ( *v2 != -4096 && *v2 != -8192 )
          break;
        v2 += 2;
        if ( v3 == v2 )
          return;
      }
      if ( v2 != v3 )
      {
        do
        {
          v5 = (_QWORD *)v4[1];
          if ( *(_DWORD *)(v5[1] + 304LL)
            || (v6 = sub_D9B120(*v5), !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v6 + 8LL))(v6)) )
          {
            v7 = *(_DWORD *)(a1 + 24);
            v8 = *(_QWORD *)(a1 + 8);
            if ( v7 )
            {
              v9 = v7 - 1;
              v10 = (v7 - 1) & (((unsigned int)*v4 >> 9) ^ ((unsigned int)*v4 >> 4));
              v11 = (__int64 *)(v8 + 16LL * v10);
              v12 = *v11;
              if ( *v4 == *v11 )
              {
LABEL_12:
                v13 = (__int64 *)v11[1];
                if ( v13 )
                  sub_D33160(v13);
                *v11 = -8192;
                --*(_DWORD *)(a1 + 16);
                ++*(_DWORD *)(a1 + 20);
              }
              else
              {
                v14 = 1;
                while ( v12 != -4096 )
                {
                  v10 = v9 & (v14 + v10);
                  v11 = (__int64 *)(v8 + 16LL * v10);
                  v12 = *v11;
                  if ( *v4 == *v11 )
                    goto LABEL_12;
                  ++v14;
                }
              }
            }
          }
          v4 += 2;
          if ( v4 == v3 )
            break;
          while ( *v4 == -4096 || *v4 == -8192 )
          {
            v4 += 2;
            if ( v3 == v4 )
              return;
          }
        }
        while ( v3 != v4 );
      }
    }
  }
}
