// Function: sub_107D670
// Address: 0x107d670
//
__int64 __fastcall sub_107D670(__int64 a1, __int64 a2, _QWORD *a3)
{
  __int64 result; // rax
  __int64 v4; // rax
  __int64 v6; // rdx
  unsigned __int64 v7; // rax
  int *v8; // r15
  int *i; // r14
  int v10; // eax
  int *v11; // rdx
  int *v12; // r14
  int *v13; // r15
  int v14; // eax
  unsigned __int64 v15; // rax
  int v16; // edi
  int v17; // r9d
  __int64 v18; // r8
  unsigned int j; // eax
  __int64 v20; // rsi
  int v21; // r10d
  unsigned int v22; // eax
  __int64 v23; // r11
  int *v24; // rdx
  _DWORD *v25; // rcx
  __int64 v26; // r11
  __int64 v27; // r11
  _DWORD *v28; // rdx
  _DWORD *v29; // rcx
  _DWORD *v30; // r11
  __int64 v31; // [rsp+8h] [rbp-D8h]
  int v33; // [rsp+1Ch] [rbp-C4h]
  int v34; // [rsp+24h] [rbp-BCh] BYREF
  __int64 v35[3]; // [rsp+28h] [rbp-B8h] BYREF
  char v36; // [rsp+40h] [rbp-A0h] BYREF
  char *v37; // [rsp+48h] [rbp-98h]
  __int64 v38; // [rsp+50h] [rbp-90h]
  char v39; // [rsp+58h] [rbp-88h] BYREF
  __int64 v40; // [rsp+68h] [rbp-78h]
  char *v41; // [rsp+70h] [rbp-70h]
  __int64 v42; // [rsp+78h] [rbp-68h]
  char v43; // [rsp+80h] [rbp-60h] BYREF
  char *v44; // [rsp+88h] [rbp-58h]
  __int64 v45; // [rsp+90h] [rbp-50h]
  char v46; // [rsp+98h] [rbp-48h] BYREF
  __int64 v47; // [rsp+A8h] [rbp-38h]

  result = *(unsigned int *)(a1 + 24);
  v33 = result;
  if ( !(_DWORD)result )
  {
    *a3 = 0;
    return result;
  }
  v4 = *(_QWORD *)(a1 + 8);
  v37 = &v39;
  v31 = v4;
  v35[1] = (__int64)&v36;
  v35[2] = 0x100000000LL;
  v40 = 0x100000000LL;
  v42 = 0x100000000LL;
  v44 = &v46;
  v38 = 0x400000000LL;
  v45 = 0x400000000LL;
  v6 = *(unsigned int *)(a2 + 60);
  v47 = 0x200000000LL;
  v41 = &v43;
  v7 = 0x9DDFEA08EB382D69LL
     * (((0x9DDFEA08EB382D69LL * ((unsigned __int64)sub_C64CA0 + 8 * v6)) >> 47)
      ^ (0x9DDFEA08EB382D69LL * ((_QWORD)sub_C64CA0 + 8 * v6)));
  v8 = *(int **)a2;
  v35[0] = 0x9DDFEA08EB382D69LL * ((v7 >> 47) ^ v7);
  for ( i = &v8[*(unsigned int *)(a2 + 8)]; i != v8; v35[0] = sub_107C450(v35, &v34) )
  {
    v10 = *v8++;
    v34 = v10;
  }
  v11 = *(int **)(a2 + 24);
  v12 = &v11[*(unsigned int *)(a2 + 32)];
  v13 = v11;
  if ( v11 == v12 )
  {
    LODWORD(v15) = v35[0];
  }
  else
  {
    do
    {
      v14 = *v13++;
      v34 = v14;
      v15 = sub_107C450(v35, &v34);
      v35[0] = v15;
    }
    while ( v12 != v13 );
  }
  v16 = *(_DWORD *)(a2 + 60);
  v17 = 1;
  v18 = 0;
  for ( j = (v33 - 1) & v15; ; j = (v33 - 1) & v22 )
  {
    v20 = v31 + 72LL * j;
    v21 = *(_DWORD *)(v20 + 60);
    if ( v16 == v21 )
    {
      v23 = *(unsigned int *)(a2 + 8);
      if ( v23 == *(_DWORD *)(v20 + 8) )
      {
        v24 = *(int **)a2;
        v25 = *(_DWORD **)v20;
        v26 = *(_QWORD *)a2 + 4 * v23;
        if ( *(_QWORD *)a2 == v26 )
        {
LABEL_20:
          v27 = *(unsigned int *)(a2 + 32);
          if ( v27 == *(_DWORD *)(v20 + 32) )
          {
            v28 = *(_DWORD **)(a2 + 24);
            v29 = *(_DWORD **)(v20 + 24);
            v30 = &v28[v27];
            if ( v28 == v30 )
            {
LABEL_24:
              *a3 = v20;
              return 1;
            }
            while ( *v28 == *v29 )
            {
              ++v28;
              ++v29;
              if ( v30 == v28 )
                goto LABEL_24;
            }
          }
        }
        else
        {
          while ( *v24 == *v25 )
          {
            ++v24;
            ++v25;
            if ( (int *)v26 == v24 )
              goto LABEL_20;
          }
        }
      }
    }
    if ( v21 == 1 )
      break;
    if ( v21 == 2 && !*(_DWORD *)(v20 + 8) && !*(_DWORD *)(v20 + 32) && !v18 )
      v18 = v31 + 72LL * j;
LABEL_15:
    v22 = v17 + j;
    ++v17;
  }
  if ( *(_DWORD *)(v20 + 8) || *(_DWORD *)(v20 + 32) )
    goto LABEL_15;
  if ( !v18 )
    v18 = v31 + 72LL * j;
  *a3 = v18;
  return 0;
}
