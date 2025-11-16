// Function: sub_1662870
// Address: 0x1662870
//
__int64 __fastcall sub_1662870(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rsi
  unsigned int v5; // ecx
  __int64 *v6; // rdx
  __int64 v7; // r8
  unsigned int v8; // r13d
  int v10; // edx
  unsigned int v11; // eax
  unsigned int v12; // esi
  __int64 v13; // rcx
  unsigned int v14; // edx
  __int64 *v15; // rax
  __int64 v16; // r8
  int v17; // r9d
  int v18; // r11d
  __int64 *v19; // r9
  int v20; // eax
  int v21; // edx
  __int64 *v22; // [rsp+8h] [rbp-88h] BYREF
  __int64 v23; // [rsp+10h] [rbp-80h] BYREF
  char v24; // [rsp+18h] [rbp-78h]
  __int64 v25; // [rsp+20h] [rbp-70h] BYREF
  _BYTE *v26; // [rsp+28h] [rbp-68h]
  _BYTE *v27; // [rsp+30h] [rbp-60h]
  __int64 v28; // [rsp+38h] [rbp-58h]
  int v29; // [rsp+40h] [rbp-50h]
  _BYTE v30[72]; // [rsp+48h] [rbp-48h] BYREF

  v3 = *(unsigned int *)(a1 + 64);
  if ( (_DWORD)v3 )
  {
    v4 = *(_QWORD *)(a1 + 48);
    v5 = (v3 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v6 = (__int64 *)(v4 + 16LL * v5);
    v7 = *v6;
    if ( a2 == *v6 )
    {
LABEL_3:
      if ( v6 != (__int64 *)(v4 + 16 * v3) )
        return *((unsigned __int8 *)v6 + 8);
    }
    else
    {
      v10 = 1;
      while ( v7 != -8 )
      {
        v17 = v10 + 1;
        v5 = (v3 - 1) & (v10 + v5);
        v6 = (__int64 *)(v4 + 16LL * v5);
        v7 = *v6;
        if ( a2 == *v6 )
          goto LABEL_3;
        v10 = v17;
      }
    }
  }
  v25 = 0;
  v26 = v30;
  v27 = v30;
  v28 = 4;
  v29 = 0;
  v11 = sub_164F8A0(a2, (__int64)&v25);
  v12 = *(_DWORD *)(a1 + 64);
  v23 = a2;
  v24 = v11;
  v8 = v11;
  if ( !v12 )
  {
    ++*(_QWORD *)(a1 + 40);
    goto LABEL_24;
  }
  v13 = *(_QWORD *)(a1 + 48);
  v14 = (v12 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v15 = (__int64 *)(v13 + 16LL * v14);
  v16 = *v15;
  if ( a2 != *v15 )
  {
    v18 = 1;
    v19 = 0;
    while ( v16 != -8 )
    {
      if ( v16 == -16 && !v19 )
        v19 = v15;
      v14 = (v12 - 1) & (v18 + v14);
      v15 = (__int64 *)(v13 + 16LL * v14);
      v16 = *v15;
      if ( a2 == *v15 )
        goto LABEL_10;
      ++v18;
    }
    if ( !v19 )
      v19 = v15;
    v20 = *(_DWORD *)(a1 + 56);
    ++*(_QWORD *)(a1 + 40);
    v21 = v20 + 1;
    if ( 4 * (v20 + 1) < 3 * v12 )
    {
      if ( v12 - *(_DWORD *)(a1 + 60) - v21 > v12 >> 3 )
      {
LABEL_20:
        *(_DWORD *)(a1 + 56) = v21;
        if ( *v19 != -8 )
          --*(_DWORD *)(a1 + 60);
        *v19 = a2;
        *((_BYTE *)v19 + 8) = v24;
        goto LABEL_10;
      }
LABEL_25:
      sub_16626B0(a1 + 40, v12);
      sub_165C500(a1 + 40, &v23, &v22);
      v19 = v22;
      a2 = v23;
      v21 = *(_DWORD *)(a1 + 56) + 1;
      goto LABEL_20;
    }
LABEL_24:
    v12 *= 2;
    goto LABEL_25;
  }
LABEL_10:
  if ( v27 != v26 )
    _libc_free((unsigned __int64)v27);
  return v8;
}
