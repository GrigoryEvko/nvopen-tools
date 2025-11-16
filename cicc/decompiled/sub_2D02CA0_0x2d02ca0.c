// Function: sub_2D02CA0
// Address: 0x2d02ca0
//
__int64 __fastcall sub_2D02CA0(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  char *v4; // rax
  _BYTE *v5; // rdi
  __int64 v6; // rax
  __int64 v7; // rdx
  size_t v8; // r15
  const void *v9; // r14
  int v10; // eax
  int v11; // eax
  __int64 v12; // rax
  __int64 v13; // rsi
  int v14; // eax
  unsigned int v15; // r10d
  __int64 *v16; // rcx
  __int64 v17; // rdx
  __int64 v18; // rax
  _BYTE *v19; // rsi
  __int64 v20; // r13
  unsigned __int8 *v21; // rax
  unsigned __int8 v22; // dl
  __int64 v23; // rax
  unsigned int v24; // r10d
  __int64 *v25; // rcx
  __int64 v26; // r8
  __int64 *v27; // rax
  __int64 *v28; // rax
  unsigned __int64 v29; // rdi
  __int64 v30; // rax
  __int64 v31; // rbx
  __int64 v32; // r12
  __int64 v34; // [rsp+8h] [rbp-78h]
  __int64 *v35; // [rsp+10h] [rbp-70h]
  __int64 v36; // [rsp+18h] [rbp-68h]
  unsigned int v37; // [rsp+18h] [rbp-68h]
  __int64 v38; // [rsp+28h] [rbp-58h] BYREF
  _BYTE *v39; // [rsp+30h] [rbp-50h] BYREF
  _BYTE *v40; // [rsp+38h] [rbp-48h]
  _BYTE *v41; // [rsp+40h] [rbp-40h]

  v2 = *(_QWORD *)(a2 + 16);
  v39 = 0;
  v40 = 0;
  v41 = 0;
  if ( !v2 )
  {
    LODWORD(v2) = 0;
    return (unsigned int)v2;
  }
  do
  {
    while ( 1 )
    {
      v20 = *(_QWORD *)(v2 + 24);
      v21 = *(unsigned __int8 **)(v20 - 32LL * (*(_DWORD *)(v20 + 4) & 0x7FFFFFF));
      v22 = *v21;
      if ( *v21 == 85 )
      {
        v21 = *(unsigned __int8 **)&v21[-32 * (*((_DWORD *)v21 + 1) & 0x7FFFFFF)];
        v22 = *v21;
      }
      if ( v22 > 0x1Cu )
      {
        if ( v22 != 63 )
          goto LABEL_4;
      }
      else if ( v22 != 5 || *((_WORD *)v21 + 1) != 34 )
      {
LABEL_4:
        if ( (v21[7] & 0x40) != 0 )
          goto LABEL_5;
        goto LABEL_26;
      }
      v21 = *(unsigned __int8 **)&v21[-32 * (*((_DWORD *)v21 + 1) & 0x7FFFFFF)];
      if ( (v21[7] & 0x40) != 0 )
      {
LABEL_5:
        v4 = (char *)*((_QWORD *)v21 - 1);
        goto LABEL_6;
      }
LABEL_26:
      v4 = (char *)&v21[-32 * (*((_DWORD *)v21 + 1) & 0x7FFFFFF)];
LABEL_6:
      v5 = *(_BYTE **)v4;
      if ( **(_BYTE **)v4 == 3 )
        v5 = (_BYTE *)*((_QWORD *)v5 - 4);
      v6 = sub_AC52D0((__int64)v5);
      v8 = v7 - 1;
      v9 = (const void *)v6;
      if ( !v7 )
        v8 = 0;
      v36 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8);
      v10 = sub_C92610();
      v11 = sub_C92860((__int64 *)a1, v9, v8, v10);
      if ( v11 == -1 )
        v12 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8);
      else
        v12 = *(_QWORD *)a1 + 8LL * v11;
      v13 = 0;
      if ( v36 != v12 )
      {
        v14 = sub_C92610();
        v15 = sub_C92740(a1, v9, v8, v14);
        v16 = (__int64 *)(*(_QWORD *)a1 + 8LL * v15);
        v17 = *v16;
        if ( *v16 )
        {
          if ( v17 != -8 )
          {
LABEL_15:
            v13 = *(int *)(v17 + 8);
            goto LABEL_16;
          }
          --*(_DWORD *)(a1 + 16);
        }
        v35 = v16;
        v37 = v15;
        v23 = sub_C7D670(v8 + 17, 8);
        v24 = v37;
        v25 = v35;
        v26 = v23;
        if ( v8 )
        {
          v34 = v23;
          memcpy((void *)(v23 + 16), v9, v8);
          v24 = v37;
          v25 = v35;
          v26 = v34;
        }
        *(_BYTE *)(v26 + v8 + 16) = 0;
        *(_QWORD *)v26 = v8;
        *(_DWORD *)(v26 + 8) = 0;
        *v25 = v26;
        ++*(_DWORD *)(a1 + 12);
        v27 = (__int64 *)(*(_QWORD *)a1 + 8LL * (unsigned int)sub_C929D0((__int64 *)a1, v24));
        v17 = *v27;
        if ( *v27 == -8 || !v17 )
        {
          v28 = v27 + 1;
          do
          {
            do
              v17 = *v28++;
            while ( !v17 );
          }
          while ( v17 == -8 );
        }
        goto LABEL_15;
      }
LABEL_16:
      v18 = sub_AD64C0(*(_QWORD *)(v20 + 8), v13, 0);
      sub_BD84D0(v20, v18);
      v38 = v20;
      v19 = v40;
      if ( v40 != v41 )
        break;
      sub_249A840((__int64)&v39, v40, &v38);
      v2 = *(_QWORD *)(v2 + 8);
      if ( !v2 )
        goto LABEL_38;
    }
    if ( v40 )
    {
      *(_QWORD *)v40 = v20;
      v19 = v40;
    }
    v40 = v19 + 8;
    v2 = *(_QWORD *)(v2 + 8);
  }
  while ( v2 );
LABEL_38:
  v29 = (unsigned __int64)v39;
  v30 = (v40 - v39) >> 3;
  if ( v40 != v39 )
  {
    if ( (_DWORD)v30 )
    {
      v31 = 0;
      v32 = 8LL * (unsigned int)(v30 - 1);
      while ( 1 )
      {
        sub_B43D60(*(_QWORD **)(v29 + v31));
        v29 = (unsigned __int64)v39;
        if ( v31 == v32 )
          break;
        v31 += 8;
      }
    }
    LODWORD(v2) = 1;
  }
  if ( v29 )
    j_j___libc_free_0(v29);
  return (unsigned int)v2;
}
