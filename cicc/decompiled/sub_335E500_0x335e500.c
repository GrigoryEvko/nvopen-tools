// Function: sub_335E500
// Address: 0x335e500
//
unsigned __int64 __fastcall sub_335E500(unsigned __int64 *a1, __int64 a2, _QWORD *a3, _DWORD *a4)
{
  unsigned __int64 v4; // rbx
  unsigned __int64 v5; // r13
  __int64 v6; // rax
  _DWORD *v7; // r9
  bool v8; // zf
  __int64 v10; // rdi
  __int64 v11; // rax
  __int64 v12; // r8
  __int64 v13; // r15
  bool v14; // cf
  unsigned __int64 v15; // rax
  unsigned __int64 v16; // rcx
  __int64 v17; // r12
  unsigned __int64 v18; // rcx
  __int64 v19; // rsi
  int v20; // edi
  unsigned __int64 v21; // r12
  __int64 v22; // rcx
  __int64 v23; // rsi
  int v24; // eax
  int v25; // ecx
  __int64 v26; // rcx
  __int64 v27; // rdx
  unsigned __int64 i; // r15
  unsigned __int64 v29; // rdi
  unsigned __int64 v30; // rdi
  unsigned __int64 v32; // r12
  __int64 v33; // rax
  _DWORD *v34; // [rsp+0h] [rbp-60h]
  _QWORD *v35; // [rsp+8h] [rbp-58h]
  unsigned __int64 v36; // [rsp+10h] [rbp-50h]
  __int64 v37; // [rsp+18h] [rbp-48h]
  __int64 v38; // [rsp+18h] [rbp-48h]
  __int64 v39; // [rsp+20h] [rbp-40h]
  __int64 v40; // [rsp+20h] [rbp-40h]
  unsigned __int64 v41; // [rsp+28h] [rbp-38h]

  v4 = a1[1];
  v5 = *a1;
  v6 = (__int64)(v4 - *a1) >> 8;
  if ( v6 == 0x7FFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v7 = a4;
  v8 = v6 == 0;
  v10 = (__int64)(a1[1] - *a1) >> 8;
  v11 = 1;
  v12 = a2;
  if ( !v8 )
    v11 = v10;
  v13 = a2;
  v14 = __CFADD__(v10, v11);
  v15 = v10 + v11;
  v16 = a2 - v5;
  if ( v14 )
  {
    v32 = 0x7FFFFFFFFFFFFF00LL;
  }
  else
  {
    if ( !v15 )
    {
      v36 = 0;
      v17 = 256;
      v41 = 0;
      goto LABEL_7;
    }
    if ( v15 > 0x7FFFFFFFFFFFFFLL )
      v15 = 0x7FFFFFFFFFFFFFLL;
    v32 = v15 << 8;
  }
  v34 = v7;
  v35 = a3;
  v33 = sub_22077B0(v32);
  v16 = a2 - v5;
  v12 = a2;
  a3 = v35;
  v7 = v34;
  v41 = v33;
  v36 = v33 + v32;
  v17 = v33 + 256;
LABEL_7:
  v18 = v41 + v16;
  if ( v18 )
  {
    v19 = *a3;
    v20 = *v7;
    *(_QWORD *)(v18 + 8) = 0;
    *(_QWORD *)(v18 + 120) = v18 + 136;
    v7 = 0;
    *(_QWORD *)v18 = v19;
    *(_QWORD *)(v18 + 40) = v18 + 56;
    *(_QWORD *)(v18 + 16) = 0;
    *(_QWORD *)(v18 + 24) = 0;
    *(_QWORD *)(v18 + 32) = 0;
    *(_QWORD *)(v18 + 48) = 0x400000000LL;
    *(_QWORD *)(v18 + 128) = 0x400000000LL;
    *(_DWORD *)(v18 + 200) = v20;
    *(_QWORD *)(v18 + 204) = 0;
    *(_QWORD *)(v18 + 212) = 0;
    *(_QWORD *)(v18 + 220) = 0;
    *(_QWORD *)(v18 + 228) = 0;
    *(_QWORD *)(v18 + 236) = 0;
    *(_QWORD *)(v18 + 244) = 0;
    *(_WORD *)(v18 + 252) = 0;
    *(_BYTE *)(v18 + 254) = 4;
  }
  if ( v12 != v5 )
  {
    v21 = v41;
    v22 = v5;
    while ( 1 )
    {
      if ( v21 )
      {
        *(_QWORD *)v21 = *(_QWORD *)v22;
        *(_QWORD *)(v21 + 8) = *(_QWORD *)(v22 + 8);
        *(_QWORD *)(v21 + 16) = *(_QWORD *)(v22 + 16);
        *(_QWORD *)(v21 + 24) = *(_QWORD *)(v22 + 24);
        v23 = *(_QWORD *)(v22 + 32);
        *(_DWORD *)(v21 + 48) = 0;
        *(_QWORD *)(v21 + 32) = v23;
        *(_QWORD *)(v21 + 40) = v21 + 56;
        *(_DWORD *)(v21 + 52) = 4;
        if ( *(_DWORD *)(v22 + 48) )
        {
          v37 = v12;
          v39 = v22;
          sub_335BB20(v21 + 40, v22 + 40, (__int64)a3, v22, v12, (__int64)v7);
          v12 = v37;
          v22 = v39;
        }
        *(_DWORD *)(v21 + 128) = 0;
        *(_QWORD *)(v21 + 120) = v21 + 136;
        *(_DWORD *)(v21 + 132) = 4;
        if ( *(_DWORD *)(v22 + 128) )
        {
          v38 = v12;
          v40 = v22;
          sub_335BB20(v21 + 120, v22 + 120, (__int64)a3, v22, v12, (__int64)v7);
          v12 = v38;
          v22 = v40;
        }
        *(_DWORD *)(v21 + 200) = *(_DWORD *)(v22 + 200);
        *(_DWORD *)(v21 + 204) = *(_DWORD *)(v22 + 204);
        *(_DWORD *)(v21 + 208) = *(_DWORD *)(v22 + 208);
        *(_DWORD *)(v21 + 212) = *(_DWORD *)(v22 + 212);
        *(_DWORD *)(v21 + 216) = *(_DWORD *)(v22 + 216);
        *(_DWORD *)(v21 + 220) = *(_DWORD *)(v22 + 220);
        *(_DWORD *)(v21 + 224) = *(_DWORD *)(v22 + 224);
        *(_DWORD *)(v21 + 228) = *(_DWORD *)(v22 + 228);
        *(_DWORD *)(v21 + 232) = *(_DWORD *)(v22 + 232);
        *(_DWORD *)(v21 + 236) = *(_DWORD *)(v22 + 236);
        *(_DWORD *)(v21 + 240) = *(_DWORD *)(v22 + 240);
        *(_DWORD *)(v21 + 244) = *(_DWORD *)(v22 + 244);
        *(_WORD *)(v21 + 248) = *(_WORD *)(v22 + 248);
        *(_WORD *)(v21 + 250) = *(_WORD *)(v22 + 250);
        *(_WORD *)(v21 + 252) = *(_WORD *)(v22 + 252);
        *(_BYTE *)(v21 + 254) = *(_BYTE *)(v22 + 254);
      }
      v22 += 256;
      if ( v12 == v22 )
        break;
      v21 += 256LL;
    }
    v17 = v21 + 512;
  }
  if ( v12 != v4 )
  {
    do
    {
      v26 = *(_QWORD *)v13;
      v27 = *(unsigned int *)(v13 + 48);
      *(_DWORD *)(v17 + 48) = 0;
      *(_DWORD *)(v17 + 52) = 4;
      *(_QWORD *)v17 = v26;
      *(_QWORD *)(v17 + 8) = *(_QWORD *)(v13 + 8);
      *(_QWORD *)(v17 + 16) = *(_QWORD *)(v13 + 16);
      *(_QWORD *)(v17 + 24) = *(_QWORD *)(v13 + 24);
      *(_QWORD *)(v17 + 32) = *(_QWORD *)(v13 + 32);
      *(_QWORD *)(v17 + 40) = v17 + 56;
      if ( (_DWORD)v27 )
        sub_335BB20(v17 + 40, v13 + 40, v27, v17 + 56, v12, (__int64)v7);
      v24 = *(_DWORD *)(v13 + 128);
      *(_DWORD *)(v17 + 128) = 0;
      *(_QWORD *)(v17 + 120) = v17 + 136;
      *(_DWORD *)(v17 + 132) = 4;
      if ( v24 )
        sub_335BB20(v17 + 120, v13 + 120, v27, v17 + 136, v12, (__int64)v7);
      v25 = *(_DWORD *)(v13 + 200);
      v13 += 256;
      v17 += 256;
      *(_DWORD *)(v17 - 56) = v25;
      *(_DWORD *)(v17 - 52) = *(_DWORD *)(v13 - 52);
      *(_DWORD *)(v17 - 48) = *(_DWORD *)(v13 - 48);
      *(_DWORD *)(v17 - 44) = *(_DWORD *)(v13 - 44);
      *(_DWORD *)(v17 - 40) = *(_DWORD *)(v13 - 40);
      *(_DWORD *)(v17 - 36) = *(_DWORD *)(v13 - 36);
      *(_DWORD *)(v17 - 32) = *(_DWORD *)(v13 - 32);
      *(_DWORD *)(v17 - 28) = *(_DWORD *)(v13 - 28);
      *(_DWORD *)(v17 - 24) = *(_DWORD *)(v13 - 24);
      *(_DWORD *)(v17 - 20) = *(_DWORD *)(v13 - 20);
      *(_DWORD *)(v17 - 16) = *(_DWORD *)(v13 - 16);
      *(_DWORD *)(v17 - 12) = *(_DWORD *)(v13 - 12);
      *(_WORD *)(v17 - 8) = *(_WORD *)(v13 - 8);
      *(_WORD *)(v17 - 6) = *(_WORD *)(v13 - 6);
      *(_WORD *)(v17 - 4) = *(_WORD *)(v13 - 4);
      *(_BYTE *)(v17 - 2) = *(_BYTE *)(v13 - 2);
    }
    while ( v4 != v13 );
  }
  for ( i = v5; i != v4; i += 256LL )
  {
    v29 = *(_QWORD *)(i + 120);
    if ( v29 != i + 136 )
      _libc_free(v29);
    v30 = *(_QWORD *)(i + 40);
    if ( v30 != i + 56 )
      _libc_free(v30);
  }
  if ( v5 )
    j_j___libc_free_0(v5);
  a1[1] = v17;
  *a1 = v41;
  a1[2] = v36;
  return v36;
}
