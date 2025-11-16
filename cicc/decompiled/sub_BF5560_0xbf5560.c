// Function: sub_BF5560
// Address: 0xbf5560
//
__int64 __fastcall sub_BF5560(__int64 a1, _BYTE *a2)
{
  __int64 v3; // rax
  __int64 v4; // rsi
  unsigned int v5; // ecx
  __int64 v6; // rdx
  _BYTE *v7; // r8
  unsigned int v8; // r13d
  int v10; // edx
  unsigned int v11; // eax
  __int64 *v12; // rsi
  __int64 v13; // rcx
  unsigned int v14; // edx
  _QWORD *v15; // rax
  _BYTE *v16; // r8
  int v17; // r9d
  int v18; // r11d
  _QWORD *v19; // r9
  int v20; // eax
  int v21; // edx
  _QWORD *v22; // [rsp+8h] [rbp-78h] BYREF
  _BYTE *v23; // [rsp+10h] [rbp-70h] BYREF
  char v24; // [rsp+18h] [rbp-68h]
  __int64 v25; // [rsp+20h] [rbp-60h] BYREF
  char *v26; // [rsp+28h] [rbp-58h]
  __int64 v27; // [rsp+30h] [rbp-50h]
  int v28; // [rsp+38h] [rbp-48h]
  char v29; // [rsp+3Ch] [rbp-44h]
  char v30; // [rsp+40h] [rbp-40h] BYREF

  v3 = *(unsigned int *)(a1 + 64);
  v4 = *(_QWORD *)(a1 + 48);
  if ( (_DWORD)v3 )
  {
    v5 = (v3 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v6 = v4 + 16LL * v5;
    v7 = *(_BYTE **)v6;
    if ( a2 == *(_BYTE **)v6 )
    {
LABEL_3:
      if ( v6 != v4 + 16 * v3 )
        return *(unsigned __int8 *)(v6 + 8);
    }
    else
    {
      v10 = 1;
      while ( v7 != (_BYTE *)-4096LL )
      {
        v17 = v10 + 1;
        v5 = (v3 - 1) & (v10 + v5);
        v6 = v4 + 16LL * v5;
        v7 = *(_BYTE **)v6;
        if ( a2 == *(_BYTE **)v6 )
          goto LABEL_3;
        v10 = v17;
      }
    }
  }
  v29 = 1;
  v25 = 0;
  v26 = &v30;
  v27 = 4;
  v28 = 0;
  v11 = sub_BDB7C0(a2, (__int64)&v25);
  v12 = (__int64 *)*(unsigned int *)(a1 + 64);
  v23 = a2;
  v24 = v11;
  v8 = v11;
  if ( !(_DWORD)v12 )
  {
    ++*(_QWORD *)(a1 + 40);
    v22 = 0;
    goto LABEL_24;
  }
  v13 = *(_QWORD *)(a1 + 48);
  v14 = ((_DWORD)v12 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v15 = (_QWORD *)(v13 + 16LL * v14);
  v16 = (_BYTE *)*v15;
  if ( a2 != (_BYTE *)*v15 )
  {
    v18 = 1;
    v19 = 0;
    while ( v16 != (_BYTE *)-4096LL )
    {
      if ( v16 == (_BYTE *)-8192LL && !v19 )
        v19 = v15;
      v14 = ((_DWORD)v12 - 1) & (v18 + v14);
      v15 = (_QWORD *)(v13 + 16LL * v14);
      v16 = (_BYTE *)*v15;
      if ( a2 == (_BYTE *)*v15 )
        goto LABEL_10;
      ++v18;
    }
    if ( !v19 )
      v19 = v15;
    v20 = *(_DWORD *)(a1 + 56);
    ++*(_QWORD *)(a1 + 40);
    v21 = v20 + 1;
    v22 = v19;
    if ( 4 * (v20 + 1) < (unsigned int)(3 * (_DWORD)v12) )
    {
      if ( (int)v12 - *(_DWORD *)(a1 + 60) - v21 > (unsigned int)v12 >> 3 )
      {
LABEL_20:
        *(_DWORD *)(a1 + 56) = v21;
        if ( *v19 != -4096 )
          --*(_DWORD *)(a1 + 60);
        *v19 = a2;
        *((_BYTE *)v19 + 8) = v24;
        goto LABEL_10;
      }
LABEL_25:
      sub_BF5380(a1 + 40, (int)v12);
      v12 = (__int64 *)&v23;
      sub_BF05B0(a1 + 40, (__int64 *)&v23, &v22);
      a2 = v23;
      v19 = v22;
      v21 = *(_DWORD *)(a1 + 56) + 1;
      goto LABEL_20;
    }
LABEL_24:
    LODWORD(v12) = 2 * (_DWORD)v12;
    goto LABEL_25;
  }
LABEL_10:
  if ( !v29 )
    _libc_free(v26, v12);
  return v8;
}
