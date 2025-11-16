// Function: sub_1A50ED0
// Address: 0x1a50ed0
//
void __fastcall sub_1A50ED0(char *a1, char *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  char *v7; // r14
  char *v8; // r13
  __int64 v9; // rbx
  char *v11; // rax
  char *v12; // r10
  char *v13; // r11
  char *v14; // rax
  __int64 v15; // r13
  char *v16; // rax
  char *v17; // r11
  char *v18; // rax
  int v19; // esi
  __int64 v20; // rdi
  int v21; // esi
  __int64 v22; // r8
  unsigned int v23; // ecx
  __int64 *v24; // rdx
  __int64 v25; // r9
  _QWORD **v26; // rdx
  _QWORD *v27; // rdx
  unsigned int i; // ecx
  __int64 v29; // r9
  unsigned int v30; // r10d
  __int64 *v31; // rdx
  __int64 v32; // rbx
  _QWORD **v33; // rdx
  _QWORD *v34; // rdx
  unsigned int j; // esi
  int v36; // edx
  int v37; // r10d
  int v38; // edx
  int v39; // r12d
  char *v41; // [rsp+10h] [rbp-50h]
  char *src; // [rsp+18h] [rbp-48h]
  __int64 v43; // [rsp+20h] [rbp-40h]
  char *v44; // [rsp+20h] [rbp-40h]
  __int64 v45; // [rsp+28h] [rbp-38h]

  if ( !a4 )
    return;
  v6 = a5;
  if ( !a5 )
    return;
  v7 = a1;
  v8 = a2;
  v9 = a4;
  if ( a4 + a5 == 2 )
  {
    v17 = a2;
    v16 = a1;
LABEL_12:
    v19 = *(_DWORD *)(a6 + 24);
    if ( !v19 )
      return;
    v20 = *(_QWORD *)v17;
    v21 = v19 - 1;
    v22 = *(_QWORD *)(a6 + 8);
    v23 = v21 & (((unsigned int)*(_QWORD *)v17 >> 9) ^ ((unsigned int)*(_QWORD *)v17 >> 4));
    v24 = (__int64 *)(v22 + 16LL * v23);
    v25 = *v24;
    if ( *(_QWORD *)v17 == *v24 )
    {
LABEL_14:
      v26 = (_QWORD **)v24[1];
      if ( v26 )
      {
        v27 = *v26;
        for ( i = 1; v27; ++i )
          v27 = (_QWORD *)*v27;
LABEL_17:
        v29 = *(_QWORD *)v16;
        v30 = v21 & (((unsigned int)*(_QWORD *)v16 >> 9) ^ ((unsigned int)*(_QWORD *)v16 >> 4));
        v31 = (__int64 *)(v22 + 16LL * v30);
        v32 = *v31;
        if ( *(_QWORD *)v16 == *v31 )
        {
LABEL_18:
          v33 = (_QWORD **)v31[1];
          if ( v33 )
          {
            v34 = *v33;
            for ( j = 1; v34; ++j )
              v34 = (_QWORD *)*v34;
            if ( j > i )
            {
              *(_QWORD *)v16 = v20;
              *(_QWORD *)v17 = v29;
            }
          }
        }
        else
        {
          v38 = 1;
          while ( v32 != -8 )
          {
            v39 = v38 + 1;
            v30 = v21 & (v38 + v30);
            v31 = (__int64 *)(v22 + 16LL * v30);
            v32 = *v31;
            if ( v29 == *v31 )
              goto LABEL_18;
            v38 = v39;
          }
        }
        return;
      }
    }
    else
    {
      v36 = 1;
      while ( v25 != -8 )
      {
        v37 = v36 + 1;
        v23 = v21 & (v36 + v23);
        v24 = (__int64 *)(v22 + 16LL * v23);
        v25 = *v24;
        if ( v20 == *v24 )
          goto LABEL_14;
        v36 = v37;
      }
    }
    i = 0;
    goto LABEL_17;
  }
  if ( a4 <= a5 )
    goto LABEL_10;
LABEL_5:
  v43 = v9 / 2;
  v11 = (char *)sub_1A504C0(v8, a3, (__int64 *)&v7[8 * (v9 / 2)], a6);
  v12 = &v7[8 * (v9 / 2)];
  v13 = v11;
  v45 = (v11 - v8) >> 3;
  while ( 1 )
  {
    v41 = v13;
    src = v12;
    v14 = sub_1A50D10(v12, v8, v13);
    v15 = v43;
    v44 = v14;
    sub_1A50ED0(v7, src, v14, v15, v45, a6);
    v6 -= v45;
    v9 -= v15;
    if ( !v9 )
      break;
    v16 = v44;
    v17 = v41;
    if ( !v6 )
      break;
    if ( v6 + v9 == 2 )
      goto LABEL_12;
    v8 = v41;
    v7 = v44;
    if ( v9 > v6 )
      goto LABEL_5;
LABEL_10:
    v45 = v6 / 2;
    v18 = (char *)sub_1A50320(v7, (__int64)v8, (__int64 *)&v8[8 * (v6 / 2)], a6);
    v13 = &v8[8 * (v6 / 2)];
    v12 = v18;
    v43 = (v18 - v7) >> 3;
  }
}
