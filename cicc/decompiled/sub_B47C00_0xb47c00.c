// Function: sub_B47C00
// Address: 0xb47c00
//
void __fastcall sub_B47C00(__int64 a1, __int64 a2, int *a3, __int64 a4)
{
  unsigned __int64 v8; // rax
  __int64 v9; // rax
  unsigned int v10; // edx
  __int64 v11; // rax
  bool v12; // zf
  _DWORD *v13; // rax
  _QWORD *v14; // rdx
  unsigned __int64 v15; // rsi
  _BYTE *v16; // r15
  _BYTE *v17; // rbx
  int v18; // ecx
  _QWORD *v19; // r9
  unsigned int v20; // r8d
  int v21; // eax
  int *v22; // rdx
  int v23; // ecx
  int v24; // edi
  _QWORD *v25; // rsi
  __int64 v26; // r12
  __int64 v27; // rcx
  __int64 v28; // r8
  __int64 v29; // r9
  int v30; // edi
  int v31; // [rsp+8h] [rbp-B8h]
  _QWORD *v32; // [rsp+18h] [rbp-A8h] BYREF
  __int64 v33; // [rsp+20h] [rbp-A0h] BYREF
  __int64 v34; // [rsp+28h] [rbp-98h]
  _DWORD *v35; // [rsp+30h] [rbp-90h] BYREF
  unsigned int v36; // [rsp+38h] [rbp-88h]
  _BYTE *v37; // [rsp+40h] [rbp-80h] BYREF
  __int64 v38; // [rsp+48h] [rbp-78h]
  _BYTE v39[112]; // [rsp+50h] [rbp-70h] BYREF

  if ( !*(_QWORD *)(a2 + 48) && (*(_BYTE *)(a2 + 7) & 0x20) == 0 )
    return;
  if ( 4 * a4 <= 0 || (4 * a4) >> 2 == 1 )
  {
    v33 = 0;
LABEL_22:
    v34 = 1;
    goto LABEL_7;
  }
  _BitScanReverse64(&v8, ((4 * a4) >> 2) - 1);
  v33 = 0;
  v9 = 1LL << (64 - ((unsigned __int8)v8 ^ 0x3Fu));
  if ( (unsigned int)v9 <= 4 )
    goto LABEL_22;
  _BitScanReverse((unsigned int *)&v9, v9 - 1);
  LOBYTE(v9) = v9 ^ 0x1F;
  v10 = 1 << (32 - v9);
  if ( v10 <= 4 )
    goto LABEL_22;
  LOBYTE(v34) = v34 & 0xFE;
  v31 = 1 << (32 - v9);
  v11 = sub_C7D670(4LL * v10, 4);
  v12 = (v34 & 1) == 0;
  v34 &= 1u;
  v35 = (_DWORD *)v11;
  v36 = v31;
  if ( v12 )
  {
    v13 = v35;
    v14 = &v35[v36];
    if ( v35 == (_DWORD *)v14 )
      goto LABEL_11;
    goto LABEL_8;
  }
LABEL_7:
  v13 = &v35;
  v14 = &v37;
  do
  {
LABEL_8:
    if ( v13 )
      *v13 = -1;
    ++v13;
  }
  while ( v13 != (_DWORD *)v14 );
LABEL_11:
  sub_B47940((__int64)&v33, a3, &a3[a4]);
  v37 = v39;
  v15 = (unsigned __int64)&v37;
  v38 = 0x400000000LL;
  sub_B9A9D0(a2, &v37);
  v16 = v37;
  v17 = &v37[16 * (unsigned int)v38];
  if ( v37 != v17 )
  {
    do
    {
      if ( a4 )
      {
        if ( (v34 & 1) != 0 )
        {
          v18 = 3;
          v19 = &v35;
        }
        else
        {
          v19 = v35;
          if ( !v36 )
            goto LABEL_16;
          v18 = v36 - 1;
        }
        v15 = v18 & (unsigned int)(37 * *(_DWORD *)v16);
        v20 = *((_DWORD *)v19 + v15);
        if ( *(_DWORD *)v16 != v20 )
        {
          v30 = 1;
          while ( v20 != -1 )
          {
            v15 = v18 & (unsigned int)(v30 + v15);
            v20 = *((_DWORD *)v19 + (unsigned int)v15);
            if ( *(_DWORD *)v16 == v20 )
              goto LABEL_15;
            ++v30;
          }
          goto LABEL_16;
        }
      }
      else
      {
        v20 = *(_DWORD *)v16;
      }
LABEL_15:
      v15 = v20;
      sub_B99FD0(a1, v20, *((_QWORD *)v16 + 1));
LABEL_16:
      v16 += 16;
    }
    while ( v17 != v16 );
  }
  if ( a4 )
  {
    if ( (v34 & 1) != 0 )
    {
      v21 = 3;
      v22 = (int *)&v35;
    }
    else
    {
      v22 = v35;
      if ( !v36 )
        goto LABEL_35;
      v21 = v36 - 1;
    }
    v23 = *v22;
    v24 = 1;
    v15 = 0;
    if ( *v22 )
    {
      while ( v23 != -1 )
      {
        v15 = v21 & (unsigned int)(v24 + v15);
        v23 = v22[(unsigned int)v15];
        if ( !v23 )
          goto LABEL_30;
        ++v24;
      }
      goto LABEL_35;
    }
  }
LABEL_30:
  v25 = *(_QWORD **)(a2 + 48);
  v32 = v25;
  if ( v25 )
  {
    v26 = a1 + 48;
    sub_B96E90(&v32, v25, 1);
    if ( !*(_QWORD *)(a1 + 48) )
      goto LABEL_33;
    goto LABEL_32;
  }
  v15 = *(_QWORD *)(a1 + 48);
  v26 = a1 + 48;
  if ( v15 )
  {
LABEL_32:
    sub_B91220(v26);
LABEL_33:
    v15 = (unsigned __int64)v32;
    *(_QWORD *)(a1 + 48) = v32;
    if ( v15 )
      sub_B976B0(&v32, v15, v26, v27, v28, v29);
  }
LABEL_35:
  if ( v37 != v39 )
    _libc_free(v37, v15);
  if ( (v34 & 1) == 0 )
    sub_C7D6A0(v35, 4LL * v36, 4);
}
