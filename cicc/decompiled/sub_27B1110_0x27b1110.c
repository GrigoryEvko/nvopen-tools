// Function: sub_27B1110
// Address: 0x27b1110
//
__int64 __fastcall sub_27B1110(__int64 a1, __int64 a2, const void **a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // r14
  char v9; // dl
  __int64 v10; // rcx
  __int64 v11; // rax
  __int64 v13; // rax
  unsigned int v14; // esi
  int v15; // eax
  __int64 *v16; // rax
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 *v20; // r14
  __int64 v21; // rdx
  _BYTE *v22; // rcx
  int v23; // eax
  _BYTE *v24; // r8
  size_t v25; // rdx
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // r8
  __int64 v29; // r9
  size_t v30; // rdx
  int v31; // eax
  int v32; // eax
  __int64 v33; // rcx
  __int64 v34; // r8
  __int64 v35; // r9
  _BYTE *v36; // [rsp+8h] [rbp-A8h]
  _BYTE *v37; // [rsp+8h] [rbp-A8h]
  __int64 v38; // [rsp+10h] [rbp-A0h] BYREF
  __int64 v39; // [rsp+18h] [rbp-98h] BYREF
  void *s2; // [rsp+20h] [rbp-90h] BYREF
  __int64 v41; // [rsp+28h] [rbp-88h]
  _BYTE v42[32]; // [rsp+30h] [rbp-80h] BYREF
  _BYTE *v43; // [rsp+50h] [rbp-60h] BYREF
  __int64 v44; // [rsp+58h] [rbp-58h]
  _BYTE v45[80]; // [rsp+60h] [rbp-50h] BYREF

  if ( (unsigned __int8)sub_27B0210(a2, a3, &v38, a4, a5, a6) )
  {
    v8 = v38;
    v9 = 0;
    v10 = *(_QWORD *)a2;
    v11 = *(_QWORD *)(a2 + 8) + 96LL * *(unsigned int *)(a2 + 24);
    goto LABEL_3;
  }
  v13 = v38;
  v14 = *(_DWORD *)(a2 + 24);
  ++*(_QWORD *)a2;
  v39 = v13;
  v15 = *(_DWORD *)(a2 + 16) + 1;
  if ( 4 * v15 >= 3 * v14 )
  {
    v14 *= 2;
    goto LABEL_24;
  }
  if ( v14 - *(_DWORD *)(a2 + 20) - v15 <= v14 >> 3 )
  {
LABEL_24:
    sub_27B06D0(a2, v14);
    sub_27B0210(a2, a3, &v39, v33, v34, v35);
    v15 = *(_DWORD *)(a2 + 16) + 1;
  }
  *(_DWORD *)(a2 + 16) = v15;
  v16 = sub_27AC980();
  s2 = v42;
  v20 = v16;
  v41 = 0x400000000LL;
  v21 = *((unsigned int *)v16 + 2);
  if ( (_DWORD)v21 )
    sub_27ABF90((__int64)&s2, (__int64)v16, v21, v17, v18, v19);
  v22 = v45;
  v44 = 0x400000000LL;
  v23 = *((_DWORD *)v20 + 14);
  v24 = v45;
  v43 = v45;
  if ( v23 )
  {
    sub_27AC1D0((__int64)&v43, (__int64)(v20 + 6), v21, (__int64)v45, (__int64)v45, v19);
    v24 = v43;
    v22 = v45;
  }
  v8 = v39;
  v25 = *(unsigned int *)(v39 + 8);
  if ( v25 != (unsigned int)v41 )
    goto LABEL_11;
  v30 = 8 * v25;
  if ( v30 )
  {
    v36 = v24;
    v31 = memcmp(*(const void **)v39, s2, v30);
    v24 = v36;
    v22 = v45;
    if ( v31 )
      goto LABEL_11;
  }
  v25 = *(unsigned int *)(v8 + 56);
  if ( v25 != (unsigned int)v44
    || (v25 *= 8LL) != 0 && (v37 = v24, v32 = memcmp(*(const void **)(v8 + 48), v24, v25), v24 = v37, v22 = v45, v32) )
  {
LABEL_11:
    --*(_DWORD *)(a2 + 20);
  }
  if ( v24 != v45 )
    _libc_free((unsigned __int64)v24);
  if ( s2 != v42 )
    _libc_free((unsigned __int64)s2);
  sub_27ABF90(v8, (__int64)a3, v25, (__int64)v22, (__int64)v24, v19);
  sub_27AC1D0(v8 + 48, (__int64)(a3 + 6), v26, v27, v28, v29);
  v10 = *(_QWORD *)a2;
  v9 = 1;
  v11 = *(_QWORD *)(a2 + 8) + 96LL * *(unsigned int *)(a2 + 24);
LABEL_3:
  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 16) = v8;
  *(_QWORD *)(a1 + 24) = v11;
  *(_QWORD *)(a1 + 8) = v10;
  *(_BYTE *)(a1 + 32) = v9;
  return a1;
}
