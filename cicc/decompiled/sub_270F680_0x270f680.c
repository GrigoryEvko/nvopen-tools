// Function: sub_270F680
// Address: 0x270f680
//
__int64 __fastcall sub_270F680(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v4; // r8
  __int64 v5; // r9
  __int64 v6; // rdx
  __int64 v7; // r15
  __int64 v8; // r14
  __int64 v9; // rdx
  __int64 v10; // r13
  unsigned __int8 *v11; // rax
  __int64 v12; // rdi
  unsigned __int8 *v13; // rbx
  __int64 v14; // rax
  __int64 v15; // rax
  int v16; // eax
  __int64 v17; // rax
  __int64 v18; // [rsp+8h] [rbp-48h]
  __int64 v19; // [rsp+10h] [rbp-40h]
  unsigned int v20; // [rsp+1Ch] [rbp-34h]

  v18 = a2;
  result = sub_AA5930(*(_QWORD *)(a1 + 40));
  v19 = v6;
  if ( result != v6 )
  {
    v7 = result;
    do
    {
      if ( v7 != a1 )
      {
        v20 = *(_DWORD *)(a1 + 4) & 0x7FFFFFF;
        if ( v20 )
        {
          v8 = 0;
          while ( 1 )
          {
            v9 = *(_QWORD *)(a1 - 8);
            v10 = *(_QWORD *)(v9 + 32LL * *(unsigned int *)(a1 + 72) + 8 * v8);
            v11 = sub_BD3990(*(unsigned __int8 **)(v9 + 32 * v8), a2);
            v12 = *(_QWORD *)(v7 - 8);
            v13 = v11;
            if ( (*(_DWORD *)(v7 + 4) & 0x7FFFFFF) != 0 )
            {
              v14 = 0;
              a2 = v12 + 32LL * *(unsigned int *)(v7 + 72);
              while ( v10 != *(_QWORD *)(a2 + 8 * v14) )
              {
                if ( (*(_DWORD *)(v7 + 4) & 0x7FFFFFF) == (_DWORD)++v14 )
                  goto LABEL_19;
              }
              v15 = 32 * v14;
            }
            else
            {
LABEL_19:
              v15 = 0x1FFFFFFFE0LL;
            }
            if ( v13 != sub_BD3990(*(unsigned __int8 **)(v12 + v15), a2) )
              break;
            v16 = ++v8;
            if ( v20 <= (unsigned int)v8 )
            {
              if ( v20 == v16 )
                goto LABEL_20;
              break;
            }
          }
        }
        else
        {
LABEL_20:
          v17 = *(unsigned int *)(v18 + 8);
          if ( v17 + 1 > (unsigned __int64)*(unsigned int *)(v18 + 12) )
          {
            a2 = v18 + 16;
            sub_C8D5F0(v18, (const void *)(v18 + 16), v17 + 1, 8u, v4, v5);
            v17 = *(unsigned int *)(v18 + 8);
          }
          *(_QWORD *)(*(_QWORD *)v18 + 8 * v17) = v7;
          ++*(_DWORD *)(v18 + 8);
          if ( !v7 )
            BUG();
        }
      }
      result = *(_QWORD *)(v7 + 32);
      if ( !result )
        BUG();
      v7 = 0;
      if ( *(_BYTE *)(result - 24) == 84 )
        v7 = result - 24;
    }
    while ( v19 != v7 );
  }
  return result;
}
