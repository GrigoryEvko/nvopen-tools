// Function: sub_F207A0
// Address: 0xf207a0
//
__int64 __fastcall sub_F207A0(__int64 a1, __int64 *a2)
{
  __int64 *v2; // r13
  __int64 v4; // r8
  __int64 v5; // r9
  __int64 *v6; // rbx
  __int64 v7; // rdx
  __int64 *v8; // r14
  __int64 *v9; // rax
  __int64 v10; // r15
  int v11; // ecx
  __int64 v12; // rdi
  int v13; // eax
  __int64 v14; // rdx
  int v15; // ecx
  unsigned int v16; // eax
  __int64 **v17; // rbx
  __int64 **v18; // rdi
  _BYTE *v19; // rdi
  _QWORD *v20; // rbx
  _BYTE *v21; // r13
  __int64 v22; // rax
  __int64 v23; // r14
  __int64 v24; // rax
  int v26; // edi
  _QWORD *v27; // [rsp+8h] [rbp-88h]
  _BYTE *v28; // [rsp+18h] [rbp-78h] BYREF
  _BYTE *v29; // [rsp+20h] [rbp-70h] BYREF
  __int64 v30; // [rsp+28h] [rbp-68h]
  _BYTE v31[96]; // [rsp+30h] [rbp-60h] BYREF

  v2 = a2;
  sub_F54ED0(a2);
  if ( (*((_BYTE *)a2 + 7) & 0x40) != 0 )
  {
    v6 = (__int64 *)*(a2 - 1);
    v7 = 32LL * (*((_DWORD *)a2 + 1) & 0x7FFFFFF);
    v8 = &v6[(unsigned __int64)v7 / 8];
  }
  else
  {
    v8 = a2;
    v7 = 32LL * (*((_DWORD *)a2 + 1) & 0x7FFFFFF);
    v6 = &a2[v7 / 0xFFFFFFFFFFFFFFF8LL];
  }
  v9 = (__int64 *)v31;
  v30 = 0x600000000LL;
  v10 = v7 >> 5;
  v11 = 0;
  v29 = v31;
  if ( (unsigned __int64)v7 > 0xC0 )
  {
    sub_C8D5F0((__int64)&v29, v31, v7 >> 5, 8u, v4, v5);
    v11 = v30;
    v9 = (__int64 *)&v29[8 * (unsigned int)v30];
  }
  if ( v6 != v8 )
  {
    do
    {
      if ( v9 )
        *v9 = *v6;
      v6 += 4;
      ++v9;
    }
    while ( v6 != v8 );
    v11 = v30;
  }
  v12 = *(_QWORD *)(a1 + 40);
  LODWORD(v30) = v11 + v10;
  sub_F0BA50(v12, (__int64)a2);
  v13 = *(_DWORD *)(a1 + 224);
  v14 = *(_QWORD *)(a1 + 208);
  if ( v13 )
  {
    v15 = v13 - 1;
    v16 = (v13 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v17 = (__int64 **)(v14 + 32LL * v16);
    a2 = *v17;
    if ( v2 == *v17 )
    {
LABEL_12:
      v18 = (__int64 **)v17[1];
      if ( v18 != v17 + 3 )
        _libc_free(v18, a2);
      *v17 = (__int64 *)-8192LL;
      --*(_DWORD *)(a1 + 216);
      ++*(_DWORD *)(a1 + 220);
    }
    else
    {
      v26 = 1;
      while ( a2 != (__int64 *)-4096LL )
      {
        v16 = v15 & (v26 + v16);
        v17 = (__int64 **)(v14 + 32LL * v16);
        a2 = *v17;
        if ( v2 == *v17 )
          goto LABEL_12;
        ++v26;
      }
    }
  }
  sub_B43D60(v2);
  v19 = v29;
  v27 = &v29[8 * (unsigned int)v30];
  if ( v27 != (_QWORD *)v29 )
  {
    v20 = v29;
    do
    {
      while ( 1 )
      {
        v21 = (_BYTE *)*v20;
        if ( *(_BYTE *)*v20 > 0x1Cu )
        {
          v22 = *(_QWORD *)(a1 + 40);
          a2 = (__int64 *)&v28;
          v28 = (_BYTE *)*v20;
          v23 = v22 + 2096;
          sub_F200C0(v22 + 2096, (__int64 *)&v28);
          v24 = *((_QWORD *)v21 + 2);
          if ( v24 )
          {
            if ( !*(_QWORD *)(v24 + 8) )
              break;
          }
        }
        if ( v27 == ++v20 )
          goto LABEL_22;
      }
      a2 = (__int64 *)&v28;
      ++v20;
      v28 = *(_BYTE **)(v24 + 24);
      sub_F200C0(v23, (__int64 *)&v28);
    }
    while ( v27 != v20 );
LABEL_22:
    v19 = v29;
  }
  *(_BYTE *)(a1 + 240) = 1;
  if ( v19 != v31 )
    _libc_free(v19, a2);
  return 0;
}
