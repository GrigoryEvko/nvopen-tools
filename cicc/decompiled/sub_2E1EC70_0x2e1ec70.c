// Function: sub_2E1EC70
// Address: 0x2e1ec70
//
__int64 __fastcall sub_2E1EC70(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r8
  __int64 *v7; // r10
  __int64 v10; // rbx
  unsigned int v11; // r13d
  __int64 *v12; // r11
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rsi
  _QWORD *v16; // rdi
  int v17; // eax
  int v18; // eax
  void **v19; // rdi
  int v20; // r12d
  __int64 v21; // rax
  __int64 v22; // rcx
  int v23; // eax
  __int64 v24; // rdx
  __int64 *v25; // rax
  __int64 *v26; // rbx
  __int64 *v27; // r15
  __int64 v28; // rdx
  __int64 result; // rax
  size_t v30; // rdx
  __int64 v32; // [rsp+8h] [rbp-E8h]
  int v33; // [rsp+20h] [rbp-D0h]
  _QWORD *v34; // [rsp+28h] [rbp-C8h]
  char v35; // [rsp+28h] [rbp-C8h]
  unsigned __int8 v36; // [rsp+28h] [rbp-C8h]
  int v37; // [rsp+3Ch] [rbp-B4h] BYREF
  __int64 v38; // [rsp+40h] [rbp-B0h] BYREF
  __int64 v39; // [rsp+48h] [rbp-A8h]
  __int64 v40; // [rsp+50h] [rbp-A0h]
  __int64 v41; // [rsp+58h] [rbp-98h]
  void **v42; // [rsp+60h] [rbp-90h]
  __int64 v43; // [rsp+68h] [rbp-88h]
  void *v44; // [rsp+70h] [rbp-80h] BYREF
  __int64 v45; // [rsp+78h] [rbp-78h]
  _BYTE v46[48]; // [rsp+80h] [rbp-70h] BYREF
  int i; // [rsp+B0h] [rbp-40h]

  v6 = a3;
  v7 = a2;
  v34 = *(_QWORD **)(a1 + 32);
  v10 = (__int64)(v34[13] - v34[12]) >> 3;
  v44 = v46;
  v11 = (unsigned int)(v10 + 63) >> 6;
  v45 = 0x600000000LL;
  if ( v11 > 6 )
  {
    sub_C8D5F0((__int64)&v44, v46, v11, 8u, a3, a6);
    memset(v44, 0, 8LL * v11);
    LODWORD(v45) = (unsigned int)(v10 + 63) >> 6;
    v7 = a2;
    v6 = a3;
  }
  else
  {
    if ( v11 )
    {
      v30 = 8LL * v11;
      if ( v30 )
      {
        v32 = v6;
        memset(v46, 0, v30);
        v7 = a2;
        v6 = v32;
      }
    }
    LODWORD(v45) = (unsigned int)(v10 + 63) >> 6;
  }
  v12 = &v7[v6];
  for ( i = v10; v7 != v12; *((_QWORD *)v44 + (*(_DWORD *)(v13 + 24) >> 6)) |= 1LL << *(_DWORD *)(v13 + 24) )
  {
    v14 = *(_QWORD *)((*v7 & 0xFFFFFFFFFFFFFFF8LL) + 16);
    if ( v14 )
    {
      v13 = *(_QWORD *)(v14 + 24);
    }
    else
    {
      v15 = *(unsigned int *)(a4 + 304);
      v16 = *(_QWORD **)(a4 + 296);
      v38 = *v7;
      v13 = *(sub_2E1D5D0(v16, (__int64)&v16[2 * v15], &v38) - 1);
    }
    ++v7;
  }
  v17 = *(_DWORD *)(v34[41] + 24LL);
  v38 = 0;
  v39 = 0;
  v33 = v17;
  v40 = 0;
  v42 = &v44;
  v18 = *(_DWORD *)(a1 + 24);
  v41 = 0;
  v37 = v18;
  v43 = 0;
  sub_2E1EA30((__int64)&v38, &v37);
  if ( (_DWORD)v43 )
  {
    v19 = v42;
    v20 = 0;
    v21 = 0;
    do
    {
      while ( 1 )
      {
        v22 = *((unsigned int *)v19 + v21);
        v23 = (*((_QWORD *)v44 + (*((_DWORD *)v19 + v21) >> 6)) >> v22) & 1;
        if ( !v23 )
        {
          if ( v33 == (_DWORD)v22 )
            goto LABEL_20;
          v24 = *(_QWORD *)(v34[12] + 8 * v22);
          v25 = *(__int64 **)(v24 + 64);
          v26 = &v25[*(unsigned int *)(v24 + 72)];
          if ( v25 != v26 )
            break;
        }
        v21 = (unsigned int)(v20 + 1);
        v20 = v21;
        if ( (_DWORD)v21 == (_DWORD)v43 )
          goto LABEL_19;
      }
      v27 = *(__int64 **)(v24 + 64);
      do
      {
        v28 = *v27++;
        v37 = *(_DWORD *)(v28 + 24);
        sub_2E1EA30((__int64)&v38, &v37);
      }
      while ( v26 != v27 );
      v21 = (unsigned int)(v20 + 1);
      v19 = v42;
      v20 = v21;
    }
    while ( (_DWORD)v21 != (_DWORD)v43 );
  }
  else
  {
    v19 = v42;
  }
LABEL_19:
  LOBYTE(v23) = 1;
LABEL_20:
  if ( v19 != &v44 )
  {
    v35 = v23;
    _libc_free((unsigned __int64)v19);
    LOBYTE(v23) = v35;
  }
  v36 = v23;
  sub_C7D6A0(v39, 4LL * (unsigned int)v41, 4);
  result = v36;
  if ( v44 != v46 )
  {
    _libc_free((unsigned __int64)v44);
    return v36;
  }
  return result;
}
