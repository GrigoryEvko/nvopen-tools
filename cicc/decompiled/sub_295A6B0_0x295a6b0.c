// Function: sub_295A6B0
// Address: 0x295a6b0
//
void __fastcall sub_295A6B0(char *a1, char *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rbx
  char *v7; // r14
  char *v8; // r13
  __int64 v9; // r12
  char *v11; // rax
  char *v12; // r11
  char *v13; // r10
  char *v14; // rax
  __int64 v15; // r13
  char *v16; // rax
  char *v17; // r10
  char *v18; // rax
  int v19; // r9d
  __int64 v20; // rdi
  __int64 v21; // r8
  __int64 v22; // rsi
  int v23; // r9d
  unsigned int v24; // ecx
  __int64 *v25; // rdx
  __int64 v26; // r11
  _QWORD **v27; // rdx
  _QWORD *v28; // rdx
  unsigned int i; // ecx
  unsigned int v30; // r11d
  __int64 *v31; // rdx
  __int64 v32; // rbx
  _QWORD **v33; // rdx
  _QWORD *v34; // rdx
  unsigned int j; // esi
  int v36; // edx
  int v37; // r12d
  int v38; // edx
  int v39; // ebx
  char *v41; // [rsp+10h] [rbp-50h]
  __int64 v42; // [rsp+18h] [rbp-48h]
  char *v43; // [rsp+18h] [rbp-48h]
  char *src; // [rsp+20h] [rbp-40h]
  __int64 v45; // [rsp+28h] [rbp-38h]

  if ( !a5 )
    return;
  v6 = a4;
  if ( !a4 )
    return;
  v7 = a1;
  v8 = a2;
  v9 = a5;
  if ( a4 + a5 == 2 )
  {
    v17 = a2;
    v16 = a1;
LABEL_12:
    v19 = *(_DWORD *)(a6 + 24);
    v20 = *(_QWORD *)v16;
    v21 = *(_QWORD *)v17;
    v22 = *(_QWORD *)(a6 + 8);
    if ( !v19 )
      return;
    v23 = v19 - 1;
    v24 = v23 & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
    v25 = (__int64 *)(v22 + 16LL * v24);
    v26 = *v25;
    if ( v21 == *v25 )
    {
LABEL_14:
      v27 = (_QWORD **)v25[1];
      if ( v27 )
      {
        v28 = *v27;
        for ( i = 1; v28; ++i )
          v28 = (_QWORD *)*v28;
LABEL_17:
        v30 = v23 & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
        v31 = (__int64 *)(v22 + 16LL * v30);
        v32 = *v31;
        if ( v20 == *v31 )
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
              *(_QWORD *)v16 = v21;
              *(_QWORD *)v17 = v20;
            }
          }
        }
        else
        {
          v36 = 1;
          while ( v32 != -4096 )
          {
            v37 = v36 + 1;
            v30 = v23 & (v36 + v30);
            v31 = (__int64 *)(v22 + 16LL * v30);
            v32 = *v31;
            if ( v20 == *v31 )
              goto LABEL_18;
            v36 = v37;
          }
        }
        return;
      }
    }
    else
    {
      v38 = 1;
      while ( v26 != -4096 )
      {
        v39 = v38 + 1;
        v24 = v23 & (v38 + v24);
        v25 = (__int64 *)(v22 + 16LL * v24);
        v26 = *v25;
        if ( v21 == *v25 )
          goto LABEL_14;
        v38 = v39;
      }
    }
    i = 0;
    goto LABEL_17;
  }
  if ( a5 >= a4 )
    goto LABEL_10;
LABEL_5:
  v42 = v6 / 2;
  v11 = (char *)sub_2958170(v8, a3, (__int64 *)&v7[8 * (v6 / 2)], a6);
  v12 = &v7[8 * (v6 / 2)];
  v13 = v11;
  v45 = (v11 - v8) >> 3;
  while ( 1 )
  {
    v41 = v13;
    src = v12;
    v14 = sub_295A4F0(v12, v8, v13);
    v15 = v42;
    v43 = v14;
    sub_295A6B0(v7, src, v14, v15, v45, a6);
    v9 -= v45;
    v6 -= v15;
    if ( !v6 )
      break;
    v16 = v43;
    v17 = v41;
    if ( !v9 )
      break;
    if ( v9 + v6 == 2 )
      goto LABEL_12;
    v8 = v41;
    v7 = v43;
    if ( v9 < v6 )
      goto LABEL_5;
LABEL_10:
    v45 = v9 / 2;
    v18 = (char *)sub_2958320(v7, (__int64)v8, (__int64 *)&v8[8 * (v9 / 2)], a6);
    v13 = &v8[8 * (v9 / 2)];
    v12 = v18;
    v42 = (v18 - v7) >> 3;
  }
}
