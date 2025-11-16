// Function: sub_9B6FD0
// Address: 0x9b6fd0
//
__int64 __fastcall sub_9B6FD0(_QWORD *a1, __int64 a2, int a3)
{
  int v4; // eax
  __int64 v5; // rdi
  int v6; // edx
  int v8; // esi
  unsigned int v9; // ecx
  int *v10; // rax
  int v11; // r8d
  char *v12; // rax
  __int64 v13; // r10
  char v14; // dl
  __int64 v15; // rsi
  char v16; // dl
  __int64 result; // rax
  __int64 v18; // rbx
  _QWORD *v19; // rsi
  _QWORD *v20; // rdx
  _QWORD *v21; // rax
  __int64 v22; // rcx
  __int64 v23; // rdi
  unsigned int v24; // r8d
  unsigned int v25; // ecx
  unsigned int v26; // edx
  unsigned int v27; // esi
  int *v28; // rax
  int v29; // r9d
  __int64 v30; // rdx
  int v31; // esi
  __int64 v32; // r9
  int v33; // esi
  unsigned int v34; // r10d
  __int64 *v35; // rax
  __int64 v36; // r11
  int v37; // eax
  int v38; // r11d
  int v39; // eax
  int v40; // r13d
  __int64 v41; // rax
  _QWORD *v42; // rax
  int v43; // eax
  int v44; // r9d
  __int64 v45; // [rsp-10h] [rbp-40h]
  __int64 v46; // [rsp-8h] [rbp-38h]
  __int64 v47; // [rsp+0h] [rbp-30h]

  v4 = *(_DWORD *)(a2 + 32);
  v5 = *(_QWORD *)(a2 + 16);
  v6 = *(_DWORD *)(a2 + 40) + a3;
  if ( !v4 )
    goto LABEL_42;
  v8 = v4 - 1;
  v9 = (v4 - 1) & (37 * v6);
  v10 = (int *)(v5 + 16LL * v9);
  v11 = *v10;
  if ( v6 != *v10 )
  {
    v43 = 1;
    while ( v11 != 0x7FFFFFFF )
    {
      v44 = v43 + 1;
      v9 = v8 & (v43 + v9);
      v10 = (int *)(v5 + 16LL * v9);
      v11 = *v10;
      if ( v6 == *v10 )
        goto LABEL_3;
      v43 = v44;
    }
LABEL_42:
    BUG();
  }
LABEL_3:
  v12 = (char *)*((_QWORD *)v10 + 1);
  LODWORD(v13) = 0;
  v14 = *v12;
  if ( (unsigned __int8)*v12 > 0x1Cu )
  {
    if ( v14 == 61 )
    {
      v13 = *((_QWORD *)v12 - 4);
      v15 = *((_QWORD *)v12 + 1);
      goto LABEL_5;
    }
    LODWORD(v13) = 0;
    if ( v14 == 62 )
      v13 = *((_QWORD *)v12 - 4);
  }
  v15 = *(_QWORD *)(*((_QWORD *)v12 - 8) + 8LL);
LABEL_5:
  v47 = sub_D34EB0(*(_QWORD *)*a1, v15, v13, *(_QWORD *)(*a1 + 8LL), a1[1], 0, 1);
  if ( !v16 || (result = 0, !v47) )
  {
    v18 = *a1;
    if ( *(_BYTE *)(v18 + 108) )
    {
      v19 = *(_QWORD **)(v18 + 88);
      v20 = &v19[*(unsigned int *)(v18 + 100)];
      v21 = v19;
      if ( v19 != v20 )
      {
        while ( a2 != *v21 )
        {
          if ( v20 == ++v21 )
            goto LABEL_13;
        }
        v22 = (unsigned int)(*(_DWORD *)(v18 + 100) - 1);
        *(_DWORD *)(v18 + 100) = v22;
        *v21 = v19[v22];
        ++*(_QWORD *)(v18 + 80);
      }
    }
    else
    {
      v42 = (_QWORD *)sub_C8CA60(v18 + 80, a2, v45, v46);
      if ( v42 )
      {
        *v42 = -2;
        ++*(_DWORD *)(v18 + 104);
        ++*(_QWORD *)(v18 + 80);
      }
    }
LABEL_13:
    v23 = *(_QWORD *)(a2 + 16);
    v24 = *(_DWORD *)(a2 + 32);
    if ( *(_DWORD *)a2 )
    {
      v25 = 0;
      do
      {
        v26 = v25 + *(_DWORD *)(a2 + 40);
        if ( v24 )
        {
          v27 = (v24 - 1) & (37 * v26);
          v28 = (int *)(v23 + 16LL * v27);
          v29 = *v28;
          if ( v26 == *v28 )
          {
LABEL_17:
            v30 = *((_QWORD *)v28 + 1);
            if ( v30 )
            {
              v31 = *(_DWORD *)(v18 + 72);
              v32 = *(_QWORD *)(v18 + 56);
              if ( v31 )
              {
                v33 = v31 - 1;
                v34 = v33 & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
                v35 = (__int64 *)(v32 + 16LL * v34);
                v36 = *v35;
                if ( *v35 == v30 )
                {
LABEL_20:
                  *v35 = -8192;
                  --*(_DWORD *)(v18 + 64);
                  ++*(_DWORD *)(v18 + 68);
                  v24 = *(_DWORD *)(a2 + 32);
                  v23 = *(_QWORD *)(a2 + 16);
                }
                else
                {
                  v39 = 1;
                  while ( v36 != -4096 )
                  {
                    v40 = v39 + 1;
                    v41 = v33 & (v34 + v39);
                    v34 = v41;
                    v35 = (__int64 *)(v32 + 16 * v41);
                    v36 = *v35;
                    if ( v30 == *v35 )
                      goto LABEL_20;
                    v39 = v40;
                  }
                }
              }
            }
          }
          else
          {
            v37 = 1;
            while ( v29 != 0x7FFFFFFF )
            {
              v38 = v37 + 1;
              v27 = (v24 - 1) & (v37 + v27);
              v28 = (int *)(v23 + 16LL * v27);
              v29 = *v28;
              if ( v26 == *v28 )
                goto LABEL_17;
              v37 = v38;
            }
          }
        }
        ++v25;
      }
      while ( *(_DWORD *)a2 > v25 );
    }
    sub_C7D6A0(v23, 16LL * v24, 8);
    j_j___libc_free_0(a2, 56);
    return 1;
  }
  return result;
}
