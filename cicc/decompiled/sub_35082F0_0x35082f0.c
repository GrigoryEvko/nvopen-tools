// Function: sub_35082F0
// Address: 0x35082f0
//
void __fastcall sub_35082F0(_QWORD **a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 v3; // rax
  unsigned int v4; // r15d
  _BYTE *v5; // rax
  unsigned __int64 v6; // rdx
  __int64 v7; // rcx
  char *j; // r8
  __int64 v9; // r9
  unsigned __int16 *v10; // r15
  unsigned __int16 v11; // ax
  unsigned int *v12; // rbx
  unsigned int *i; // r14
  char *v14; // rax
  __int64 v15; // r10
  __int64 v16; // rsi
  unsigned int v17; // eax
  unsigned __int16 *v18; // rax
  unsigned __int16 *v19; // rbx
  unsigned __int16 *v20; // r14
  unsigned int v21; // esi
  __int64 v22; // rdx
  __int64 v23; // rcx
  unsigned __int16 *v24; // rbx
  __int64 v25; // r8
  __int64 v26; // r9
  unsigned __int16 v27; // ax
  unsigned int *v28; // r12
  unsigned int *k; // rbx
  char *v30; // rax
  __int64 v31; // rdx
  unsigned __int16 *v32; // r10
  char *m; // r9
  _QWORD *v34; // r8
  __int64 v35; // rsi
  unsigned int v36; // eax
  _QWORD *v37; // rcx
  char *v38; // rdx
  char *v39; // rax
  _QWORD *v40; // [rsp+0h] [rbp-70h] BYREF
  unsigned __int16 *v41; // [rsp+8h] [rbp-68h]
  __int64 v42; // [rsp+10h] [rbp-60h]
  __int64 v43; // [rsp+18h] [rbp-58h]
  _BYTE v44[16]; // [rsp+20h] [rbp-50h] BYREF
  _BYTE *v45; // [rsp+30h] [rbp-40h]
  unsigned int v46; // [rsp+38h] [rbp-38h]

  v2 = *(_QWORD *)(a2 + 48);
  if ( *(_BYTE *)(v2 + 120) )
  {
    if ( a1[2] )
    {
      v3 = (__int64)*a1;
      v42 = 0;
      v43 = 8;
      v45 = 0;
      v46 = 0;
      v4 = *(_DWORD *)(v3 + 16);
      v40 = (_QWORD *)v3;
      v41 = (unsigned __int16 *)v44;
      if ( v4 )
      {
        v5 = _libc_calloc(v4, 1u);
        if ( !v5 )
          sub_C64F00("Allocation failed", 1u);
        v45 = v5;
        v46 = v4;
      }
      v10 = sub_2EBFBC0(*(_QWORD **)(a2 + 32));
      if ( v10 )
      {
        while ( 1 )
        {
          v11 = *v10;
          if ( !*v10 )
            break;
          ++v10;
          sub_3507B80(&v40, v11, v6, v7, (__int64)j, v9);
        }
      }
      v12 = *(unsigned int **)(v2 + 104);
      for ( i = *(unsigned int **)(v2 + 96); v12 != i; i += 3 )
      {
        v14 = sub_E922F0(v40, *i);
        v9 = (__int64)&v14[2 * v6];
        for ( j = v14; (char *)v9 != j; j += 2 )
        {
          v6 = (unsigned __int64)v45;
          v15 = v42;
          v16 = *(unsigned __int16 *)j;
          v17 = (unsigned __int8)v45[v16];
          if ( v17 < (unsigned int)v42 )
          {
            v7 = (__int64)v41;
            while ( 1 )
            {
              v6 = (unsigned __int64)&v41[v17];
              if ( (_WORD)v16 == *(_WORD *)v6 )
                break;
              v17 += 256;
              if ( (unsigned int)v42 <= v17 )
                goto LABEL_20;
            }
            if ( (unsigned __int16 *)v6 != &v41[v42] )
            {
              v18 = &v41[v42 - 1];
              if ( (unsigned __int16 *)v6 != v18 )
              {
                *(_WORD *)v6 = *v18;
                v7 = v41[v42 - 1];
                v6 = (__int64)(v6 - (_QWORD)v41) >> 1;
                v45[v7] = v6;
                v15 = v42;
              }
              v42 = v15 - 1;
            }
          }
LABEL_20:
          ;
        }
      }
      v19 = v41;
      v20 = &v41[v42];
      if ( v20 != v41 )
      {
        do
        {
          v21 = *v19++;
          sub_3507B80(a1, v21, v6, v7, (__int64)j, v9);
        }
        while ( v20 != v19 );
      }
      if ( v45 )
        _libc_free((unsigned __int64)v45);
      if ( v41 != (unsigned __int16 *)v44 )
        _libc_free((unsigned __int64)v41);
    }
    else
    {
      v24 = sub_2EBFBC0(*(_QWORD **)(a2 + 32));
      if ( v24 )
      {
        while ( 1 )
        {
          v27 = *v24;
          if ( !*v24 )
            break;
          ++v24;
          sub_3507B80(a1, v27, v22, v23, v25, v26);
        }
      }
      v28 = *(unsigned int **)(v2 + 96);
      for ( k = *(unsigned int **)(v2 + 104); k != v28; v28 += 3 )
      {
        v30 = sub_E922F0(*a1, *v28);
        v32 = (unsigned __int16 *)&v30[2 * v31];
        for ( m = v30; v32 != (unsigned __int16 *)m; m += 2 )
        {
          v34 = a1[2];
          v35 = *(unsigned __int16 *)m;
          v36 = *((unsigned __int8 *)a1[6] + v35);
          if ( v36 < (unsigned int)v34 )
          {
            v37 = a1[1];
            while ( 1 )
            {
              v38 = (char *)v37 + 2 * v36;
              if ( (_WORD)v35 == *(_WORD *)v38 )
                break;
              v36 += 256;
              if ( (unsigned int)v34 <= v36 )
                goto LABEL_43;
            }
            if ( v38 != (char *)v37 + 2 * (_QWORD)v34 )
            {
              v39 = (char *)v37 + 2 * (_QWORD)v34 - 2;
              if ( v38 != v39 )
              {
                *(_WORD *)v38 = *(_WORD *)v39;
                *((_BYTE *)a1[6] + *((unsigned __int16 *)a1[1] + (_QWORD)a1[2] - 1)) = (v38 - (char *)a1[1]) >> 1;
                v34 = a1[2];
              }
              a1[2] = (_QWORD *)((char *)v34 - 1);
            }
          }
LABEL_43:
          ;
        }
      }
    }
  }
}
