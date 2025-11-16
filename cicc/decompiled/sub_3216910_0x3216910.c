// Function: sub_3216910
// Address: 0x3216910
//
_QWORD *__fastcall sub_3216910(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int64 a5, __int64 a6)
{
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  _QWORD *v12; // rax
  _QWORD *v13; // r12
  __int64 *v15; // rdi
  __int64 v16; // rax
  unsigned __int64 v17; // r12
  __int64 v18; // rax
  void *v19; // rdi
  int v20; // r10d
  _BYTE *v21; // rsi
  size_t v22; // rdx
  __int64 *v23; // rsi
  __int64 *v24; // rsi
  __int64 *v25; // rdx
  int v26; // [rsp+8h] [rbp-1B8h]
  int v27; // [rsp+8h] [rbp-1B8h]
  __int64 *v28; // [rsp+10h] [rbp-1B0h] BYREF
  unsigned __int64 v29; // [rsp+18h] [rbp-1A8h] BYREF
  unsigned __int64 v30[2]; // [rsp+20h] [rbp-1A0h] BYREF
  _BYTE v31[128]; // [rsp+30h] [rbp-190h] BYREF
  __int64 v32; // [rsp+B0h] [rbp-110h] BYREF
  int v33; // [rsp+B8h] [rbp-108h]
  __int16 v34; // [rsp+BCh] [rbp-104h]
  char v35; // [rsp+BEh] [rbp-102h]
  _BYTE *v36; // [rsp+C0h] [rbp-100h] BYREF
  __int64 v37; // [rsp+C8h] [rbp-F8h]
  _BYTE src[240]; // [rsp+D0h] [rbp-F0h] BYREF

  v30[1] = 0x2000000000LL;
  v30[0] = (unsigned __int64)v31;
  sub_3214F00((__int64)&v32, a2, a3, a4, a5, a6);
  sub_3214BB0((__int64)&v32, (__int64)v30, v8, v9, v10, v11);
  v12 = sub_C65B40((__int64)(a1 + 1), (__int64)v30, (__int64 *)&v28, (__int64)off_4A35610);
  if ( v12 )
  {
    v13 = v12;
    *(_DWORD *)(a2 + 24) = *((_DWORD *)v12 + 2);
    goto LABEL_3;
  }
  v15 = (__int64 *)*a1;
  v16 = *(_QWORD *)*a1;
  v15[10] += 224;
  v17 = (v16 + 15) & 0xFFFFFFFFFFFFFFF0LL;
  if ( v15[1] < v17 + 224 || !v16 )
  {
    v17 = sub_9D1E70((__int64)v15, 224, 224, 4);
    goto LABEL_11;
  }
  *v15 = v17 + 224;
  if ( v17 )
  {
LABEL_11:
    v18 = v32;
    v19 = (void *)(v17 + 32);
    *(_QWORD *)(v17 + 16) = v17 + 32;
    *(_QWORD *)v17 = v18;
    *(_DWORD *)(v17 + 8) = v33;
    *(_WORD *)(v17 + 12) = v34;
    *(_BYTE *)(v17 + 14) = v35;
    *(_QWORD *)(v17 + 24) = 0xC00000000LL;
    v20 = v37;
    if ( (_BYTE **)(v17 + 16) != &v36 && (_DWORD)v37 )
    {
      if ( v36 == src )
      {
        v21 = src;
        v22 = 16LL * (unsigned int)v37;
        if ( (unsigned int)v37 <= 0xC
          || (v27 = v37,
              sub_C8D5F0(v17 + 16, (const void *)(v17 + 32), (unsigned int)v37, 0x10u, (unsigned int)v37, (__int64)src),
              v19 = *(void **)(v17 + 16),
              v21 = v36,
              v20 = v27,
              (v22 = 16LL * (unsigned int)v37) != 0) )
        {
          v26 = v20;
          memcpy(v19, v21, v22);
          v20 = v26;
        }
        *(_DWORD *)(v17 + 24) = v20;
        LODWORD(v37) = 0;
      }
      else
      {
        *(_QWORD *)(v17 + 16) = v36;
        v36 = src;
        *(_QWORD *)(v17 + 24) = v37;
        v37 = 0;
      }
    }
  }
  v29 = v17;
  v23 = (__int64 *)a1[4];
  if ( v23 == (__int64 *)a1[5] )
  {
    sub_3216780((__int64)(a1 + 3), v23, &v29);
    v17 = v29;
    v24 = (__int64 *)a1[4];
  }
  else
  {
    if ( v23 )
    {
      *v23 = v17;
      v23 = (__int64 *)a1[4];
    }
    v24 = v23 + 1;
    a1[4] = (__int64)v24;
  }
  v25 = v28;
  *(_DWORD *)(v17 + 8) = ((__int64)v24 - a1[3]) >> 3;
  *(_DWORD *)(a2 + 24) = (a1[4] - a1[3]) >> 3;
  sub_C657C0(a1 + 1, (__int64 *)v17, v25, (__int64)off_4A35610);
  v13 = (_QWORD *)v29;
LABEL_3:
  if ( v36 != src )
    _libc_free((unsigned __int64)v36);
  if ( (_BYTE *)v30[0] != v31 )
    _libc_free(v30[0]);
  return v13;
}
