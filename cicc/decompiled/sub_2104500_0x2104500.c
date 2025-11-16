// Function: sub_2104500
// Address: 0x2104500
//
void __fastcall sub_2104500(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  unsigned int v3; // edx
  _QWORD *v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rax
  unsigned int v7; // r14d
  unsigned __int64 v8; // rcx
  unsigned int *v9; // r9
  unsigned int *i; // r8
  __int64 v11; // r10
  int v12; // eax
  _WORD *v13; // rdx
  _WORD *v14; // rsi
  _WORD *v15; // rdx
  int v16; // eax
  unsigned int v17; // esi
  __int64 v18; // rax
  __int64 v19; // rsi
  _DWORD *v20; // r9
  _DWORD *j; // r8
  __int64 v22; // rsi
  int v23; // eax
  _WORD *v24; // rdx
  _WORD *v25; // rsi
  unsigned __int64 v26; // rcx
  _WORD *v27; // rdx
  _QWORD *v28; // rax
  int v29; // eax
  unsigned __int64 v30; // r15
  __int64 v31; // rax
  unsigned __int64 v32; // r9
  char *v33; // r10
  unsigned __int64 v34; // rdx
  unsigned int v35; // r15d
  unsigned __int64 v36; // rax
  unsigned int v37; // ecx
  unsigned int v38; // eax
  unsigned __int64 v39; // rcx
  unsigned __int64 v40; // rcx
  int v41; // r14d
  __int64 v42; // rax
  unsigned __int64 v43; // [rsp+0h] [rbp-70h]
  unsigned int v44; // [rsp+8h] [rbp-68h]
  char *s; // [rsp+10h] [rbp-60h]
  unsigned int v46; // [rsp+1Ch] [rbp-54h]
  __int64 v47; // [rsp+20h] [rbp-50h] BYREF
  char *v48; // [rsp+28h] [rbp-48h]
  unsigned __int64 v49; // [rsp+30h] [rbp-40h]
  unsigned int v50; // [rsp+38h] [rbp-38h]

  v2 = *(_QWORD *)(a2 + 56);
  if ( !*(_BYTE *)(v2 + 104) )
    return;
  v3 = (unsigned int)(*(_DWORD *)(a1 + 24) + 63) >> 6;
  if ( v3 )
  {
    v4 = *(_QWORD **)(a1 + 8);
    v5 = (__int64)&v4[v3];
    while ( !*v4 )
    {
      if ( (_QWORD *)v5 == ++v4 )
        goto LABEL_20;
    }
    v6 = *(_QWORD *)a1;
    v48 = 0;
    v49 = 0;
    v7 = *(_DWORD *)(v6 + 44);
    v50 = 0;
    v47 = v6;
    if ( !v7 )
      goto LABEL_7;
    v46 = v7 + 63;
    v30 = (v7 + 63) >> 6;
    v31 = malloc(8 * v30);
    v32 = v30;
    v33 = (char *)v31;
    if ( v31 )
      goto LABEL_29;
    if ( 8 * v30 )
    {
      sub_16BD1C0("Allocation failed", 1u);
      v33 = 0;
      v32 = (v7 + 63) >> 6;
      v35 = v50 & 0x3F;
      v37 = (v50 + 63) >> 6;
      v36 = v37;
    }
    else
    {
      v42 = malloc(1u);
      v32 = (v7 + 63) >> 6;
      v33 = (char *)v42;
      if ( v42 )
      {
LABEL_29:
        v48 = v33;
        v49 = v32;
        if ( !(_DWORD)v30 )
        {
LABEL_34:
          v38 = v50;
          if ( v7 > v50 )
          {
            v39 = (v50 + 63) >> 6;
            if ( v49 > v39 && v49 != v39 )
            {
              memset(&v48[8 * v39], 0, 8 * (v49 - v39));
              v38 = v50;
            }
            if ( (v38 & 0x3F) != 0 )
            {
              *(_QWORD *)&v48[8 * ((v50 + 63) >> 6) - 8] &= ~(-1LL << (v38 & 0x3F));
              v38 = v50;
            }
          }
          v50 = v7;
          if ( v7 < v38 )
          {
            v40 = v46 >> 6;
            if ( v49 > v40 && v49 != v40 )
            {
              memset(&v48[8 * v40], 0, 8 * (v49 - v40));
              LOBYTE(v7) = v50;
            }
            v41 = v7 & 0x3F;
            if ( v41 )
              *(_QWORD *)&v48[8 * (v46 >> 6) - 8] &= ~(-1LL << v41);
          }
LABEL_7:
          sub_2103DC0(&v47, *(_QWORD **)(a2 + 40));
          v9 = *(unsigned int **)(v2 + 88);
          for ( i = *(unsigned int **)(v2 + 80); v9 != i; i += 3 )
          {
            if ( !v47 )
              BUG();
            v11 = *i;
            v12 = *(_DWORD *)(*(_QWORD *)(v47 + 8) + 24 * v11 + 16) & 0xF;
            v13 = (_WORD *)(*(_QWORD *)(v47 + 56) + 2LL * (*(_DWORD *)(*(_QWORD *)(v47 + 8) + 24 * v11 + 16) >> 4));
            v8 = (unsigned int)(v11 * v12);
            v14 = v13 + 1;
            LOWORD(v8) = *v13 + v11 * v12;
LABEL_11:
            v15 = v14;
            while ( v15 )
            {
              ++v15;
              *(_QWORD *)&v48[(v8 >> 3) & 0x1FF8] &= ~(1LL << v8);
              v16 = (unsigned __int16)*(v15 - 1);
              v14 = 0;
              v8 = (unsigned int)(v16 + v8);
              if ( !(_WORD)v16 )
                goto LABEL_11;
            }
          }
          v17 = v50;
          if ( *(_DWORD *)(a1 + 24) < v50 )
          {
            sub_13A49F0(a1 + 8, v50, 0, v8, (int)i, (int)v9);
            v17 = v50;
          }
          v18 = 0;
          v19 = (v17 + 63) >> 6;
          if ( (_DWORD)v19 )
          {
            do
            {
              *(_QWORD *)(*(_QWORD *)(a1 + 8) + 8 * v18) |= *(_QWORD *)&v48[8 * v18];
              ++v18;
            }
            while ( v19 != v18 );
          }
          _libc_free((unsigned __int64)v48);
          return;
        }
        v34 = v32;
        v35 = 0;
        v36 = 0;
        v37 = 0;
        goto LABEL_31;
      }
      sub_16BD1C0("Allocation failed", 1u);
      v32 = (v7 + 63) >> 6;
      v33 = 0;
      v35 = v50 & 0x3F;
      v37 = (v50 + 63) >> 6;
      v36 = v37;
    }
    v48 = 0;
    v49 = v32;
    if ( v36 >= v32 )
    {
      if ( !v35 )
        goto LABEL_48;
      goto LABEL_47;
    }
    v34 = v32 - v36;
    if ( v32 == v36 )
    {
LABEL_32:
      if ( !v35 )
      {
LABEL_33:
        memset(v33, 0, 8 * v32);
        goto LABEL_34;
      }
LABEL_47:
      *(_QWORD *)&v33[8 * v37 - 8] &= ~(-1LL << v35);
      v32 = v49;
      v33 = v48;
LABEL_48:
      if ( !v32 )
        goto LABEL_34;
      goto LABEL_33;
    }
LABEL_31:
    v43 = v32;
    v44 = v37;
    s = v33;
    memset(&v33[8 * v36], 0, 8 * v34);
    v32 = v43;
    v37 = v44;
    v33 = s;
    goto LABEL_32;
  }
LABEL_20:
  sub_2103DC0((__int64 *)a1, *(_QWORD **)(a2 + 40));
  v20 = *(_DWORD **)(v2 + 88);
  for ( j = *(_DWORD **)(v2 + 80); v20 != j; j += 3 )
  {
    v22 = *(_QWORD *)a1;
    if ( !*(_QWORD *)a1 )
      BUG();
    v23 = *(_DWORD *)(*(_QWORD *)(v22 + 8) + 24LL * (unsigned int)*j + 16) & 0xF;
    v24 = (_WORD *)(*(_QWORD *)(v22 + 56) + 2LL
                                          * (*(_DWORD *)(*(_QWORD *)(v22 + 8) + 24LL * (unsigned int)*j + 16) >> 4));
    v26 = (unsigned int)(*j * v23);
    v25 = v24 + 1;
    LOWORD(v26) = *v24 + *j * v23;
    while ( 1 )
    {
      v27 = v25;
      if ( !v25 )
        break;
      while ( 1 )
      {
        ++v27;
        v28 = (_QWORD *)(*(_QWORD *)(a1 + 8) + ((v26 >> 3) & 0x1FF8));
        *v28 &= ~(1LL << v26);
        v29 = (unsigned __int16)*(v27 - 1);
        v25 = 0;
        if ( !(_WORD)v29 )
          break;
        v26 = (unsigned int)(v29 + v26);
        if ( !v27 )
          goto LABEL_26;
      }
    }
LABEL_26:
    ;
  }
}
