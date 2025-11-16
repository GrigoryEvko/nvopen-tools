// Function: sub_2579F40
// Address: 0x2579f40
//
__int64 __fastcall sub_2579F40(__int64 a1)
{
  __int64 v2; // rax
  __int64 v3; // rdi
  __int64 v4; // r8
  __int64 v5; // r9
  _BYTE *v6; // rax
  unsigned __int64 v7; // rdx
  unsigned __int64 v8; // rcx
  int v9; // r15d
  _BYTE *v10; // rsi
  _BYTE *v11; // rdi
  __int64 v12; // rdx
  char v13; // al
  unsigned __int64 v14; // r12
  unsigned __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  char v20; // al
  unsigned __int64 v22; // rdi
  __int64 v23; // rax
  __int64 v24; // rsi
  _QWORD *v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // rdx
  _BYTE *v28; // rax
  __int64 v29; // r8
  __int64 v30; // rsi
  void *v31; // [rsp+0h] [rbp-140h] BYREF
  void *v32; // [rsp+8h] [rbp-138h]
  __int16 v33; // [rsp+10h] [rbp-130h]
  __int64 v34; // [rsp+18h] [rbp-128h]
  __int64 v35; // [rsp+20h] [rbp-120h]
  __int64 v36; // [rsp+28h] [rbp-118h]
  __int64 v37; // [rsp+30h] [rbp-110h]
  _BYTE *v38; // [rsp+38h] [rbp-108h]
  __int64 v39; // [rsp+40h] [rbp-100h]
  _BYTE src[192]; // [rsp+48h] [rbp-F8h] BYREF
  char v41; // [rsp+108h] [rbp-38h]

  *(_WORD *)(a1 + 104) = 256;
  v34 = 0;
  v41 = 0;
  v31 = &unk_4A171B8;
  v33 = 256;
  v35 = 0;
  v38 = src;
  v32 = &unk_4A16CD8;
  v39 = 0x800000000LL;
  v2 = *(unsigned int *)(a1 + 136);
  v36 = 0;
  v3 = *(_QWORD *)(a1 + 120);
  v37 = 0;
  sub_C7D6A0(v3, 24 * v2, 8);
  ++*(_QWORD *)(a1 + 112);
  ++v34;
  *(_QWORD *)(a1 + 120) = 0;
  v35 = 0;
  *(_QWORD *)(a1 + 128) = v36;
  v36 = 0;
  *(_DWORD *)(a1 + 136) = v37;
  v6 = v38;
  LODWORD(v37) = 0;
  if ( v38 == src )
  {
    v7 = (unsigned int)v39;
    v8 = *(unsigned int *)(a1 + 152);
    v9 = v39;
    if ( (unsigned int)v39 <= v8 )
    {
      v11 = src;
      if ( !(_DWORD)v39 )
      {
LABEL_8:
        v13 = v41;
        *(_DWORD *)(a1 + 152) = v9;
        *(_BYTE *)(a1 + 352) = v13;
        v31 = &unk_4A171B8;
        if ( v11 != src )
          _libc_free((unsigned __int64)v11);
        goto LABEL_10;
      }
      v23 = *(_QWORD *)(a1 + 144);
      v24 = v23 + 24LL * (unsigned int)v39;
      v25 = src;
      do
      {
        v26 = *v25;
        v23 += 24;
        v25 += 3;
        *(_QWORD *)(v23 - 24) = v26;
        *(_QWORD *)(v23 - 16) = *(v25 - 2);
        *(_BYTE *)(v23 - 8) = *((_BYTE *)v25 - 8);
      }
      while ( v23 != v24 );
    }
    else
    {
      if ( (unsigned int)v39 > (unsigned __int64)*(unsigned int *)(a1 + 156) )
      {
        *(_DWORD *)(a1 + 152) = 0;
        sub_C8D5F0(a1 + 144, (const void *)(a1 + 160), v7, 0x18u, v4, v5);
        v11 = v38;
        v7 = (unsigned int)v39;
        v8 = 0;
        v10 = v38;
      }
      else
      {
        v10 = src;
        v11 = src;
        if ( *(_DWORD *)(a1 + 152) )
        {
          v27 = *(_QWORD *)(a1 + 144);
          v28 = src;
          v29 = 24 * v8;
          v8 *= 24LL;
          do
          {
            v30 = *(_QWORD *)v28;
            v28 += 24;
            v27 += 24;
            *(_QWORD *)(v27 - 24) = v30;
            *(_QWORD *)(v27 - 16) = *((_QWORD *)v28 - 2);
            *(_BYTE *)(v27 - 8) = *(v28 - 8);
          }
          while ( v28 != &src[v29] );
          v11 = v38;
          v7 = (unsigned int)v39;
          v10 = &v38[v29];
        }
      }
      v12 = 24 * v7;
      if ( v10 == &v11[v12] )
        goto LABEL_8;
      memcpy((void *)(v8 + *(_QWORD *)(a1 + 144)), v10, v12 - v8);
    }
    v11 = v38;
    goto LABEL_8;
  }
  v22 = *(_QWORD *)(a1 + 144);
  if ( v22 != a1 + 160 )
  {
    _libc_free(v22);
    v6 = v38;
  }
  *(_QWORD *)(a1 + 144) = v6;
  *(_QWORD *)(a1 + 152) = v39;
  *(_BYTE *)(a1 + 352) = v41;
  v31 = &unk_4A171B8;
LABEL_10:
  sub_C7D6A0(v35, 24LL * (unsigned int)v37, 8);
  v14 = sub_250D070((_QWORD *)(a1 + 72));
  v15 = sub_2509740((_QWORD *)(a1 + 72));
  v31 = (void *)v14;
  v32 = (void *)v15;
  v20 = *(_BYTE *)(a1 + 105);
  LOBYTE(v33) = 3;
  if ( v20 )
  {
    sub_2579B00(a1 + 88, (__int64)&v31, v16, v17, v18, v19);
    v20 = *(_BYTE *)(a1 + 105);
  }
  *(_BYTE *)(a1 + 104) = v20;
  return 0;
}
