// Function: sub_16B0990
// Address: 0x16b0990
//
void __fastcall sub_16B0990(__int64 **a1, __int64 a2, char a3)
{
  int v4; // ecx
  __int64 *v5; // rsi
  __int64 v7; // rax
  __int64 *v8; // rax
  __int64 v9; // rdx
  __int64 *v10; // r15
  __int64 *v11; // r14
  __int64 v12; // rsi
  int v13; // eax
  _QWORD *v14; // rax
  char v15; // dl
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // rax
  __int64 *v19; // rax
  __int64 v20; // rdx
  __int64 *v21; // rax
  __int64 v22; // rdx
  _QWORD *v23; // rdi
  _QWORD *v24; // rcx
  __int64 v25; // [rsp+0h] [rbp-170h]
  __int64 v26; // [rsp+8h] [rbp-168h]
  __int64 v27; // [rsp+10h] [rbp-160h] BYREF
  _BYTE *v28; // [rsp+18h] [rbp-158h]
  _BYTE *v29; // [rsp+20h] [rbp-150h]
  __int64 v30; // [rsp+28h] [rbp-148h]
  int v31; // [rsp+30h] [rbp-140h]
  _BYTE v32[312]; // [rsp+38h] [rbp-138h] BYREF

  v4 = *((_DWORD *)a1 + 2);
  v27 = 0;
  v28 = v32;
  v29 = v32;
  v30 = 32;
  v31 = 0;
  if ( v4 )
  {
    v5 = *a1;
    v7 = **a1;
    if ( v7 != -8 && v7 )
    {
      v10 = *a1;
    }
    else
    {
      v8 = v5 + 1;
      do
      {
        do
        {
          v9 = *v8;
          v10 = v8++;
        }
        while ( v9 == -8 );
      }
      while ( !v9 );
    }
    v11 = &v5[v4];
    if ( v11 != v10 )
    {
      while ( 1 )
      {
        v12 = *(_QWORD *)(*v10 + 8);
        v13 = (*(_BYTE *)(v12 + 12) >> 5) & 3;
        if ( v13 == 2 || v13 == 1 && !a3 )
          goto LABEL_16;
        v14 = v28;
        if ( v29 != v28 )
          goto LABEL_12;
        v23 = &v28[8 * HIDWORD(v30)];
        if ( v28 == (_BYTE *)v23 )
          break;
        v24 = 0;
        while ( v12 != *v14 )
        {
          if ( *v14 == -2 )
            v24 = v14;
          if ( v23 == ++v14 )
          {
            if ( !v24 )
              goto LABEL_35;
            *v24 = v12;
            --v31;
            ++v27;
            goto LABEL_13;
          }
        }
LABEL_16:
        v20 = v10[1];
        v21 = v10 + 1;
        if ( !v20 || v20 == -8 )
        {
          do
          {
            do
            {
              v22 = v21[1];
              ++v21;
            }
            while ( v22 == -8 );
          }
          while ( !v22 );
        }
        if ( v21 == v11 )
          goto LABEL_22;
        v10 = v21;
      }
LABEL_35:
      if ( HIDWORD(v30) >= (unsigned int)v30 )
      {
LABEL_12:
        sub_16CCBA0(&v27, v12);
        if ( !v15 )
          goto LABEL_16;
      }
      else
      {
        ++HIDWORD(v30);
        *v23 = v12;
        ++v27;
      }
LABEL_13:
      v16 = *(_QWORD *)(*v10 + 8);
      v17 = *v10 + 16;
      v18 = *(unsigned int *)(a2 + 8);
      if ( (unsigned int)v18 >= *(_DWORD *)(a2 + 12) )
      {
        v25 = *(_QWORD *)(*v10 + 8);
        v26 = *v10 + 16;
        sub_16CD150(a2, a2 + 16, 0, 16);
        v18 = *(unsigned int *)(a2 + 8);
        v16 = v25;
        v17 = v26;
      }
      v19 = (__int64 *)(*(_QWORD *)a2 + 16 * v18);
      *v19 = v17;
      v19[1] = v16;
      ++*(_DWORD *)(a2 + 8);
      goto LABEL_16;
    }
  }
LABEL_22:
  if ( *(unsigned int *)(a2 + 8) > 1uLL )
    qsort(*(void **)a2, (16LL * *(unsigned int *)(a2 + 8)) >> 4, 0x10u, (__compar_fn_t)sub_16B0340);
  if ( v29 != v28 )
    _libc_free((unsigned __int64)v29);
}
