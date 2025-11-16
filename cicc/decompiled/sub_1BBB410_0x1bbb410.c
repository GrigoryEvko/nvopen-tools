// Function: sub_1BBB410
// Address: 0x1bbb410
//
void __fastcall sub_1BBB410(__int64 *src, __int64 *a2, __int64 a3)
{
  __int64 *v3; // r9
  __int64 *v5; // r15
  __int64 v7; // rbx
  __int64 *i; // r12
  __int64 *v9; // rdx
  __int64 v10; // r13
  __int64 v11; // rbx
  __int64 v12; // rdi
  bool v13; // dl
  __int64 v14; // r10
  unsigned int v15; // eax
  __int64 v16; // rax
  __int64 *v17; // rbx
  unsigned int v18; // eax
  __int64 v19; // rax
  __int64 v20; // [rsp-50h] [rbp-50h]
  __int64 v21; // [rsp-50h] [rbp-50h]
  __int64 *v22; // [rsp-48h] [rbp-48h]
  __int64 *v23; // [rsp-48h] [rbp-48h]
  __int64 v24; // [rsp-48h] [rbp-48h]
  __int64 v25; // [rsp-40h] [rbp-40h]
  __int64 *v26; // [rsp-40h] [rbp-40h]

  if ( src != a2 )
  {
    v3 = a2;
    v5 = src + 1;
    if ( a2 != src + 1 )
    {
      while ( 1 )
      {
        while ( 1 )
        {
          v10 = *v5;
          v11 = *src;
          i = v5;
          v12 = *(_QWORD *)(a3 + 1352);
          v13 = *v5 == 0;
          if ( *src != 0 && !v13 && *v5 != *src )
            break;
LABEL_4:
          v7 = *(v5 - 1);
          if ( !v13 && v7 != 0 && v7 != v10 )
          {
            for ( i = v5 - 1; ; --i )
            {
              v9 = i + 1;
              if ( *(_QWORD *)(v7 + 8) == v10 )
                goto LABEL_34;
              if ( *(_QWORD *)(v10 + 8) == v7 || *(_DWORD *)(v10 + 16) >= *(_DWORD *)(v7 + 16) )
                goto LABEL_13;
              if ( *(_BYTE *)(v12 + 72) )
                break;
              v18 = *(_DWORD *)(v12 + 76) + 1;
              *(_DWORD *)(v12 + 76) = v18;
              if ( v18 > 0x20 )
              {
                v21 = a3;
                v23 = v3;
                sub_15CC640(v12);
                v9 = i + 1;
                v3 = v23;
                a3 = v21;
                if ( *(_DWORD *)(v7 + 48) < *(_DWORD *)(v10 + 48) )
                  goto LABEL_13;
                goto LABEL_12;
              }
              do
              {
                v19 = v7;
                v7 = *(_QWORD *)(v7 + 8);
              }
              while ( v7 && *(_DWORD *)(v10 + 16) <= *(_DWORD *)(v7 + 16) );
              if ( v19 != v10 )
              {
LABEL_13:
                i = v9;
                goto LABEL_14;
              }
LABEL_34:
              v7 = *(i - 1);
              i[1] = *i;
              v12 = *(_QWORD *)(a3 + 1352);
              if ( !v7 || v7 == v10 )
                goto LABEL_14;
            }
            if ( *(_DWORD *)(v7 + 48) < *(_DWORD *)(v10 + 48) )
              goto LABEL_13;
LABEL_12:
            if ( *(_DWORD *)(v7 + 52) > *(_DWORD *)(v10 + 52) )
              goto LABEL_13;
            goto LABEL_34;
          }
LABEL_14:
          *i = v10;
          if ( v3 == ++v5 )
            return;
        }
        v14 = *(_QWORD *)(v11 + 8);
        if ( v10 == v14 )
          goto LABEL_25;
        if ( v11 == *(_QWORD *)(v10 + 8) || *(_DWORD *)(v10 + 16) >= *(_DWORD *)(v11 + 16) )
          goto LABEL_38;
        if ( *(_BYTE *)(v12 + 72) )
          break;
        v15 = *(_DWORD *)(v12 + 76) + 1;
        *(_DWORD *)(v12 + 76) = v15;
        if ( v15 <= 0x20 )
        {
          do
          {
            v16 = v11;
            v11 = *(_QWORD *)(v11 + 8);
          }
          while ( v11 && *(_DWORD *)(v10 + 16) <= *(_DWORD *)(v11 + 16) );
          v14 = *v5;
          if ( v10 == v16 )
            goto LABEL_25;
LABEL_47:
          v12 = *(_QWORD *)(a3 + 1352);
          v10 = v14;
          v13 = v14 == 0;
          goto LABEL_4;
        }
        v24 = a3;
        v26 = v3;
        sub_15CC640(v12);
        v3 = v26;
        a3 = v24;
        if ( *(_DWORD *)(v11 + 48) < *(_DWORD *)(v10 + 48) )
        {
          v10 = *v5;
          v12 = *(_QWORD *)(v24 + 1352);
          v13 = *v5 == 0;
          goto LABEL_4;
        }
        v14 = *v5;
        if ( *(_DWORD *)(v11 + 52) > *(_DWORD *)(v10 + 52) )
          goto LABEL_47;
LABEL_25:
        v17 = v5 + 1;
        if ( src != v5 )
        {
          v20 = a3;
          v22 = v3;
          v25 = v14;
          memmove(src + 1, src, (char *)v5 - (char *)src);
          a3 = v20;
          v3 = v22;
          v14 = v25;
        }
        *src = v14;
        ++v5;
        if ( v3 == v17 )
          return;
      }
      if ( *(_DWORD *)(v11 + 48) >= *(_DWORD *)(v10 + 48) && *(_DWORD *)(v11 + 52) <= *(_DWORD *)(v10 + 52) )
      {
        v14 = *v5;
        goto LABEL_25;
      }
LABEL_38:
      v13 = 0;
      goto LABEL_4;
    }
  }
}
