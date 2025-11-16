// Function: sub_D24710
// Address: 0xd24710
//
__int64 __fastcall sub_D24710(__int64 a1, __int64 a2, void (__fastcall *a3)(__int64, __int64), __int64 a4, __int64 a5)
{
  __int64 result; // rax
  unsigned __int64 v6; // rcx
  __int64 v7; // r12
  __int64 *v8; // rdx
  __int64 *v9; // r9
  __int64 *v10; // r13
  __int64 v11; // rbx
  __int64 *v12; // rax
  __int64 v13; // rax

  for ( result = *(unsigned int *)(a1 + 8); (_DWORD)result; result = *(unsigned int *)(a1 + 8) )
  {
    while ( 1 )
    {
      v6 = (unsigned int)result;
      result = (unsigned int)(result - 1);
      v7 = *(_QWORD *)(*(_QWORD *)a1 + 8 * v6 - 8);
      *(_DWORD *)(a1 + 8) = result;
      if ( !*(_BYTE *)v7 )
        break;
      if ( *(_BYTE *)v7 != 4 )
      {
        v8 = (__int64 *)(32LL * (*(_DWORD *)(v7 + 4) & 0x7FFFFFF));
        if ( (*(_BYTE *)(v7 + 7) & 0x40) != 0 )
        {
          v9 = *(__int64 **)(v7 - 8);
          v7 = (__int64)v8 + (_QWORD)v9;
        }
        else
        {
          v9 = (__int64 *)(v7 - (_QWORD)v8);
        }
        if ( v9 != (__int64 *)v7 )
        {
          v10 = v9;
          v11 = *v9;
          if ( *(_BYTE *)(a2 + 28) )
          {
LABEL_14:
            v12 = *(__int64 **)(a2 + 8);
            v6 = *(unsigned int *)(a2 + 20);
            v8 = &v12[v6];
            if ( v12 == v8 )
              goto LABEL_26;
            while ( v11 != *v12 )
            {
              if ( v8 == ++v12 )
              {
LABEL_26:
                if ( (unsigned int)v6 < *(_DWORD *)(a2 + 16) )
                {
                  *(_DWORD *)(a2 + 20) = v6 + 1;
                  *v8 = v11;
                  ++*(_QWORD *)a2;
                  goto LABEL_21;
                }
                goto LABEL_20;
              }
            }
            goto LABEL_18;
          }
          while ( 1 )
          {
LABEL_20:
            sub_C8CC70(a2, v11, (__int64)v8, v6, a5, (__int64)v9);
            if ( (_BYTE)v8 )
            {
LABEL_21:
              v13 = *(unsigned int *)(a1 + 8);
              v6 = *(unsigned int *)(a1 + 12);
              if ( v13 + 1 > v6 )
              {
                sub_C8D5F0(a1, (const void *)(a1 + 16), v13 + 1, 8u, a5, (__int64)v9);
                v13 = *(unsigned int *)(a1 + 8);
              }
              v8 = *(__int64 **)a1;
              v10 += 4;
              *(_QWORD *)(*(_QWORD *)a1 + 8 * v13) = v11;
              ++*(_DWORD *)(a1 + 8);
              if ( (__int64 *)v7 == v10 )
                goto LABEL_7;
            }
            else
            {
LABEL_18:
              v10 += 4;
              if ( (__int64 *)v7 == v10 )
                goto LABEL_7;
            }
            v11 = *v10;
            if ( *(_BYTE *)(a2 + 28) )
              goto LABEL_14;
          }
        }
      }
LABEL_4:
      if ( !(_DWORD)result )
        return result;
    }
    if ( !sub_B2FC80(v7) )
    {
      a3(a4, v7);
      result = *(unsigned int *)(a1 + 8);
      goto LABEL_4;
    }
LABEL_7:
    ;
  }
  return result;
}
