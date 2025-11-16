// Function: sub_16D3060
// Address: 0x16d3060
//
__int64 __fastcall sub_16D3060(char *a1, unsigned __int64 a2, char *a3, unsigned __int64 a4, char a5, unsigned int a6)
{
  unsigned __int64 v7; // r13
  __int64 v10; // rdi
  _DWORD *v11; // r15
  unsigned int v12; // eax
  __int64 v13; // rdx
  unsigned int *v14; // r8
  unsigned __int64 v15; // r13
  unsigned int v16; // esi
  unsigned int v17; // ecx
  char *v18; // rdi
  char v19; // r11
  unsigned int *v20; // rax
  unsigned int v21; // r9d
  unsigned int v22; // r9d
  unsigned int v23; // edx
  char v24; // r10
  unsigned int v25; // edx
  unsigned int v26; // r12d
  __int64 v28; // [rsp+8h] [rbp-158h]
  unsigned int v32; // [rsp+30h] [rbp-130h] BYREF

  v7 = a4 + 1;
  if ( a4 + 1 <= 0x40 )
  {
    v28 = 0;
    v11 = &v32;
    if ( !a4 )
    {
      if ( !a2 )
        return v32;
      goto LABEL_8;
    }
  }
  else
  {
    v10 = 4 * v7;
    if ( v7 > 0x1FFFFFFFFFFFFFFELL )
      v10 = -1;
    v28 = sub_2207820(v10);
    v11 = (_DWORD *)v28;
  }
  v12 = 1;
  v13 = 1;
  do
  {
    v11[v13] = v12++;
    v13 = v12;
  }
  while ( v12 <= a4 );
  if ( a2 )
  {
LABEL_8:
    v14 = &v11[v7];
    v15 = 1;
    while ( 1 )
    {
      *v11 = v15;
      v16 = v15;
      v17 = v15 - 1;
      if ( a4 )
      {
        v18 = a3;
        v19 = *a1;
        v20 = v11 + 1;
        do
        {
          v23 = v17;
          v24 = *v18;
          v17 = *v20;
          if ( a5 )
          {
            v21 = *v20;
            if ( *(v20 - 1) <= v17 )
              v21 = *(v20 - 1);
            v22 = v21 + 1;
            v23 += v19 != v24;
            if ( v22 <= v23 )
              v23 = v22;
            *v20 = v23;
          }
          else
          {
            if ( v19 != v24 )
            {
              v25 = *v20;
              if ( *(v20 - 1) <= v17 )
                v25 = *(v20 - 1);
              v23 = v25 + 1;
            }
            *v20 = v23;
          }
          if ( v16 > v23 )
            v16 = v23;
          ++v20;
          ++v18;
        }
        while ( v14 != v20 );
      }
      if ( a6 && a6 < v16 )
        break;
      ++v15;
      ++a1;
      if ( a2 < v15 )
        goto LABEL_28;
    }
    v26 = a6 + 1;
    goto LABEL_29;
  }
LABEL_28:
  v26 = v11[a4];
LABEL_29:
  if ( v28 )
    j_j___libc_free_0_0(v28);
  return v26;
}
