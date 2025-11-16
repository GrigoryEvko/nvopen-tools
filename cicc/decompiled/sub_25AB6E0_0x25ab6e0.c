// Function: sub_25AB6E0
// Address: 0x25ab6e0
//
void __fastcall sub_25AB6E0(__int64 a1, unsigned __int8 *a2)
{
  __int64 v2; // r14
  unsigned int v3; // eax
  __int64 v4; // r12
  __int64 v5; // rbx
  __int64 i; // r13
  unsigned __int8 **v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  unsigned __int8 **v11; // rax

  if ( a1 )
  {
    v2 = *(_QWORD *)(a1 - 32);
    v3 = *(_DWORD *)(v2 + 4) & 0x7FFFFFF;
    if ( v3 )
    {
      v4 = (__int64)a2;
      v5 = v3 - 1;
      for ( i = 0; ; ++i )
      {
        a2 = sub_BD3990(*(unsigned __int8 **)(v2 + 32 * (i - v3)), (__int64)a2);
        if ( *(_BYTE *)(v4 + 28) )
        {
          v11 = *(unsigned __int8 ***)(v4 + 8);
          v8 = *(unsigned int *)(v4 + 20);
          v7 = &v11[v8];
          if ( v11 != v7 )
          {
            while ( a2 != *v11 )
            {
              if ( v7 == ++v11 )
                goto LABEL_13;
            }
LABEL_9:
            if ( v5 == i )
              return;
            goto LABEL_10;
          }
LABEL_13:
          if ( (unsigned int)v8 < *(_DWORD *)(v4 + 16) )
          {
            *(_DWORD *)(v4 + 20) = v8 + 1;
            *v7 = a2;
            ++*(_QWORD *)v4;
            goto LABEL_9;
          }
        }
        sub_C8CC70(v4, (__int64)a2, (__int64)v7, v8, v9, v10);
        if ( v5 == i )
          return;
LABEL_10:
        v3 = *(_DWORD *)(v2 + 4) & 0x7FFFFFF;
      }
    }
  }
}
