// Function: sub_39833A0
// Address: 0x39833a0
//
_QWORD *__fastcall sub_39833A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v8; // rax
  _QWORD *v9; // r12
  __int64 v11; // r12
  __int64 v12; // rax
  unsigned int v13; // r9d
  void *v14; // rdi
  _BYTE *v15; // rsi
  _BYTE *v16; // rsi
  __int64 *v17; // rdx
  int v18; // eax
  _BYTE *v19; // rsi
  size_t v20; // rdx
  unsigned int v21; // [rsp+8h] [rbp-1B8h]
  unsigned int v22; // [rsp+8h] [rbp-1B8h]
  __int64 *v23; // [rsp+10h] [rbp-1B0h] BYREF
  __int64 v24; // [rsp+18h] [rbp-1A8h] BYREF
  unsigned __int64 v25[2]; // [rsp+20h] [rbp-1A0h] BYREF
  _BYTE v26[128]; // [rsp+30h] [rbp-190h] BYREF
  __int64 v27; // [rsp+B0h] [rbp-110h] BYREF
  int v28; // [rsp+B8h] [rbp-108h]
  __int16 v29; // [rsp+BCh] [rbp-104h]
  char v30; // [rsp+BEh] [rbp-102h]
  _BYTE *v31; // [rsp+C0h] [rbp-100h]
  __int64 v32; // [rsp+C8h] [rbp-F8h]
  _BYTE src[240]; // [rsp+D0h] [rbp-F0h] BYREF

  v25[1] = 0x2000000000LL;
  v25[0] = (unsigned __int64)v26;
  sub_3981CE0((__int64)&v27, a2, a3, a4, a5, a6);
  sub_3981A60((__int64)&v27, (__int64)v25);
  v8 = sub_16BDDE0(a1 + 8, (__int64)v25, (__int64 *)&v23);
  if ( v8 )
  {
    v9 = v8;
    *(_DWORD *)(a2 + 24) = *((_DWORD *)v8 + 2);
  }
  else
  {
    v11 = sub_145CDC0(0xE0u, *(__int64 **)a1);
    if ( v11 )
    {
      v12 = v27;
      v13 = v32;
      v14 = (void *)(v11 + 32);
      *(_QWORD *)(v11 + 16) = v11 + 32;
      *(_QWORD *)v11 = v12;
      *(_DWORD *)(v11 + 8) = v28;
      *(_WORD *)(v11 + 12) = v29;
      *(_BYTE *)(v11 + 14) = v30;
      *(_QWORD *)(v11 + 24) = 0xC00000000LL;
      if ( v13 )
      {
        if ( v31 == src )
        {
          v19 = src;
          v20 = 16LL * v13;
          if ( v13 <= 0xC
            || (v22 = v13,
                sub_16CD150(v11 + 16, (const void *)(v11 + 32), v13, 16, (int)src, v13),
                v14 = *(void **)(v11 + 16),
                v19 = v31,
                v13 = v22,
                (v20 = 16LL * (unsigned int)v32) != 0) )
          {
            v21 = v13;
            memcpy(v14, v19, v20);
            v13 = v21;
          }
          *(_DWORD *)(v11 + 24) = v13;
          LODWORD(v32) = 0;
        }
        else
        {
          *(_QWORD *)(v11 + 16) = v31;
          v18 = HIDWORD(v32);
          *(_DWORD *)(v11 + 24) = v13;
          *(_DWORD *)(v11 + 28) = v18;
          v31 = src;
          v32 = 0;
        }
      }
    }
    v24 = v11;
    v15 = *(_BYTE **)(a1 + 40);
    if ( v15 == *(_BYTE **)(a1 + 48) )
    {
      sub_3983210(a1 + 32, v15, &v24);
      v11 = v24;
      v16 = *(_BYTE **)(a1 + 40);
    }
    else
    {
      if ( v15 )
      {
        *(_QWORD *)v15 = v11;
        v15 = *(_BYTE **)(a1 + 40);
      }
      v16 = v15 + 8;
      *(_QWORD *)(a1 + 40) = v16;
    }
    v17 = v23;
    *(_DWORD *)(v11 + 8) = (__int64)&v16[-*(_QWORD *)(a1 + 32)] >> 3;
    *(_DWORD *)(a2 + 24) = (__int64)(*(_QWORD *)(a1 + 40) - *(_QWORD *)(a1 + 32)) >> 3;
    sub_16BDA20((__int64 *)(a1 + 8), (__int64 *)v11, v17);
    v9 = (_QWORD *)v24;
  }
  if ( v31 != src )
    _libc_free((unsigned __int64)v31);
  if ( (_BYTE *)v25[0] != v26 )
    _libc_free(v25[0]);
  return v9;
}
