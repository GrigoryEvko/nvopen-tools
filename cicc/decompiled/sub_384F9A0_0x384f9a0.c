// Function: sub_384F9A0
// Address: 0x384f9a0
//
__int64 __fastcall sub_384F9A0(__int64 a1, __int64 a2)
{
  __int64 v2; // r8
  __int64 *v3; // r13
  __int64 **v4; // r10
  __int64 v5; // rax
  char *v6; // rbx
  char *v7; // rax
  __int64 v8; // rax
  unsigned __int64 v9; // rdx
  unsigned __int64 v10; // r12
  __int64 *v11; // rcx
  int v12; // esi
  __int64 *v13; // r11
  __int64 *v14; // rax
  __int64 v15; // rsi
  int v16; // eax
  __int64 v17; // r8
  int v18; // ebx
  __int64 result; // rax
  __int64 v20; // rax
  __int64 *v21; // rbx
  int v22; // edx
  __int64 v23; // rsi
  __int64 v25; // [rsp+10h] [rbp-90h]
  __int64 **v26; // [rsp+18h] [rbp-88h]
  unsigned __int64 v27; // [rsp+20h] [rbp-80h]
  __int64 v28; // [rsp+28h] [rbp-78h]
  __int64 v29; // [rsp+38h] [rbp-68h] BYREF
  __int64 *v30; // [rsp+40h] [rbp-60h] BYREF
  __int64 v31; // [rsp+48h] [rbp-58h]
  __int64 v32[10]; // [rsp+50h] [rbp-50h] BYREF

  v2 = a2;
  v3 = (__int64 *)a2;
  v4 = *(__int64 ***)a1;
  v5 = 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
  {
    v6 = *(char **)(a2 - 8);
    v7 = &v6[v5];
  }
  else
  {
    v6 = (char *)(a2 - v5);
    v7 = (char *)a2;
  }
  v8 = v7 - v6;
  v30 = v32;
  v31 = 0x400000000LL;
  v9 = 0xAAAAAAAAAAAAAAABLL * (v8 >> 3);
  v10 = v9;
  if ( (unsigned __int64)v8 > 0x60 )
  {
    v27 = 0xAAAAAAAAAAAAAAABLL * (v8 >> 3);
    v25 = v8;
    v26 = v4;
    sub_16CD150((__int64)&v30, v32, v9, 8, a2, (int)v32);
    v13 = v30;
    v12 = v31;
    LODWORD(v9) = v27;
    v4 = v26;
    v8 = v25;
    v11 = &v30[(unsigned int)v31];
    v2 = a2;
  }
  else
  {
    v11 = v32;
    v12 = 0;
    v13 = v32;
  }
  if ( v8 > 0 )
  {
    v14 = (__int64 *)v6;
    do
    {
      v15 = *v14;
      ++v11;
      v14 += 3;
      *(v11 - 1) = v15;
      --v10;
    }
    while ( v10 );
    v13 = v30;
    v12 = v31;
  }
  LODWORD(v31) = v12 + v9;
  v28 = v2;
  v16 = sub_14A5330(v4, v2, (__int64)v13, (unsigned int)(v12 + v9));
  v17 = v28;
  v18 = v16;
  if ( v30 != v32 )
  {
    _libc_free((unsigned __int64)v30);
    v17 = v28;
  }
  result = 1;
  if ( v18 )
  {
    v20 = 3LL * (*(_DWORD *)(v17 + 20) & 0xFFFFFFF);
    v21 = (__int64 *)(v17 - v20 * 8);
    if ( (*(_BYTE *)(v17 + 23) & 0x40) != 0 )
    {
      v21 = *(__int64 **)(v17 - 8);
      v3 = &v21[v20];
    }
    for ( ; v21 != v3; v21 += 3 )
    {
      v22 = *(_DWORD *)(a1 + 184);
      v23 = *v21;
      v30 = 0;
      v31 = -1;
      v32[0] = 0;
      v32[1] = 0;
      if ( v22 && *(_DWORD *)(a1 + 216) && sub_384F1D0(a1, v23, &v29, &v30) )
        sub_384F170(a1, v32[0]);
    }
    return 0;
  }
  return result;
}
