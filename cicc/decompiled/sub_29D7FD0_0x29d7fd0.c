// Function: sub_29D7FD0
// Address: 0x29d7fd0
//
__int64 __fastcall sub_29D7FD0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v5; // r12
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // r12
  unsigned __int64 v9; // rsi
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rbx
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rbx
  __int64 v17; // rax
  __int64 v18; // rbx
  __int64 v19; // rax
  __int64 v20; // rax
  size_t *v21; // rcx
  unsigned __int64 v22; // r12
  __int64 v23; // rax
  size_t **v24; // rax
  size_t v25; // r10
  size_t v26; // r9
  unsigned __int64 v27; // r8
  size_t v28; // rdx
  int v29; // eax
  unsigned int v30; // eax
  __int64 v31; // [rsp+8h] [rbp-58h]
  unsigned __int64 v32; // [rsp+10h] [rbp-50h]
  size_t v33; // [rsp+18h] [rbp-48h]
  size_t *v34; // [rsp+20h] [rbp-40h]
  size_t v35; // [rsp+20h] [rbp-40h]
  unsigned int v36; // [rsp+2Ch] [rbp-34h]

  v5 = 0;
  if ( *(char *)(a3 + 7) < 0 )
  {
    v6 = sub_BD2BC0(a3);
    v8 = v6 + v7;
    if ( *(char *)(a3 + 7) < 0 )
      v8 -= sub_BD2BC0(a3);
    v5 = (unsigned int)(v8 >> 4);
  }
  v9 = 0;
  if ( *(char *)(a2 + 7) < 0 )
  {
    v10 = sub_BD2BC0(a2);
    v12 = v10 + v11;
    if ( *(char *)(a2 + 7) >= 0 )
      v9 = (unsigned int)(v12 >> 4);
    else
      v9 = (unsigned int)((v12 - sub_BD2BC0(a2)) >> 4);
  }
  v36 = sub_29D7CF0(a1, v9, v5);
  if ( !v36 && *(char *)(a2 + 7) < 0 )
  {
    v14 = sub_BD2BC0(a2);
    v16 = v14 + v15;
    if ( *(char *)(a2 + 7) >= 0 )
      v17 = v16 >> 4;
    else
      LODWORD(v17) = (v16 - sub_BD2BC0(a2)) >> 4;
    if ( (_DWORD)v17 )
    {
      v18 = 0;
      v31 = 16LL * (unsigned int)v17;
      while ( 1 )
      {
        v19 = 0;
        if ( *(char *)(a2 + 7) < 0 )
          v19 = sub_BD2BC0(a2);
        v20 = v18 + v19;
        v21 = *(size_t **)v20;
        v22 = *(unsigned int *)(v20 + 12) - (unsigned __int64)*(unsigned int *)(v20 + 8);
        v23 = 0;
        if ( *(char *)(a3 + 7) < 0 )
        {
          v34 = v21;
          v23 = sub_BD2BC0(a3);
          v21 = v34;
        }
        v24 = (size_t **)(v18 + v23);
        v25 = *v21;
        v26 = **v24;
        v27 = *((unsigned int *)v24 + 3) - (unsigned __int64)*((unsigned int *)v24 + 2);
        v28 = v26;
        if ( *v21 <= v26 )
          v28 = *v21;
        if ( v28 )
        {
          v32 = *((unsigned int *)v24 + 3) - (unsigned __int64)*((unsigned int *)v24 + 2);
          v33 = **v24;
          v35 = *v21;
          v29 = memcmp(v21 + 2, *v24 + 2, v28);
          v25 = v35;
          v26 = v33;
          v27 = v32;
          if ( v29 )
          {
            if ( v29 < 0 )
              return (unsigned int)-1;
            return 1;
          }
        }
        if ( v25 != v26 )
          break;
        v30 = sub_29D7CF0(a1, v22, v27);
        if ( v30 )
          return v30;
        v18 += 16;
        if ( v18 == v31 )
          return v36;
      }
      if ( v25 < v26 )
        return (unsigned int)-1;
      return 1;
    }
  }
  return v36;
}
