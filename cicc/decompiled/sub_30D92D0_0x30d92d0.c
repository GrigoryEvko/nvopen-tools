// Function: sub_30D92D0
// Address: 0x30d92d0
//
__int64 __fastcall sub_30D92D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v8; // r8d
  __int64 v9; // r8
  _QWORD *v10; // rbx
  __int64 v11; // r8
  __int64 v12; // rax
  unsigned __int64 v13; // rdx
  _BYTE *v14; // r12
  int v15; // ecx
  __int64 v16; // rsi
  int v17; // ecx
  unsigned int v18; // edx
  _QWORD *v19; // rax
  _BYTE *v20; // rdi
  unsigned int v21; // r12d
  __int64 v23; // rbx
  int v24; // eax
  __int64 v25; // rdx
  __int64 *v26; // rsi
  __int64 v27; // [rsp+8h] [rbp-88h]
  __int64 v28; // [rsp+18h] [rbp-78h] BYREF
  __int64 *v29; // [rsp+20h] [rbp-70h] BYREF
  __int64 v30; // [rsp+28h] [rbp-68h]
  _BYTE v31[96]; // [rsp+30h] [rbp-60h] BYREF

  v8 = *(_DWORD *)(a2 + 4);
  v29 = (__int64 *)v31;
  v30 = 0x600000000LL;
  v9 = 4LL * (v8 & 0x7FFFFFF);
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
  {
    v10 = *(_QWORD **)(a2 - 8);
    v11 = (__int64)&v10[v9];
  }
  else
  {
    v10 = (_QWORD *)(a2 - v9 * 8);
    v11 = a2;
  }
  if ( v10 == (_QWORD *)v11 )
  {
    v25 = 0;
    v26 = (__int64 *)v31;
  }
  else
  {
    do
    {
      v14 = (_BYTE *)*v10;
      if ( *(_BYTE *)*v10 > 0x15u )
      {
        v15 = *(_DWORD *)(a1 + 160);
        v16 = *(_QWORD *)(a1 + 144);
        if ( !v15 )
          goto LABEL_12;
        v17 = v15 - 1;
        v18 = v17 & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
        v19 = (_QWORD *)(v16 + 16LL * v18);
        v20 = (_BYTE *)*v19;
        if ( v14 != (_BYTE *)*v19 )
        {
          v24 = 1;
          while ( v20 != (_BYTE *)-4096LL )
          {
            a6 = (unsigned int)(v24 + 1);
            v18 = v17 & (v24 + v18);
            v19 = (_QWORD *)(v16 + 16LL * v18);
            v20 = (_BYTE *)*v19;
            if ( v14 == (_BYTE *)*v19 )
              goto LABEL_11;
            v24 = a6;
          }
LABEL_12:
          v21 = 0;
          goto LABEL_13;
        }
LABEL_11:
        v14 = (_BYTE *)v19[1];
        if ( !v14 )
          goto LABEL_12;
      }
      v12 = (unsigned int)v30;
      v13 = (unsigned int)v30 + 1LL;
      if ( v13 > HIDWORD(v30) )
      {
        v27 = v11;
        sub_C8D5F0((__int64)&v29, v31, v13, 8u, v11, a6);
        v12 = (unsigned int)v30;
        v11 = v27;
      }
      v10 += 4;
      v29[v12] = (__int64)v14;
      v25 = (unsigned int)(v30 + 1);
      LODWORD(v30) = v30 + 1;
    }
    while ( (_QWORD *)v11 != v10 );
    v26 = v29;
  }
  v21 = 0;
  v23 = sub_97D230((unsigned __int8 *)a2, v26, v25, *(_BYTE **)(a1 + 80), 0, 1u);
  if ( v23 )
  {
    v28 = a2;
    v21 = 1;
    *sub_30D9190(a1 + 136, &v28) = v23;
  }
LABEL_13:
  if ( v29 != (__int64 *)v31 )
    _libc_free((unsigned __int64)v29);
  return v21;
}
