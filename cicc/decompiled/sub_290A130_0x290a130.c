// Function: sub_290A130
// Address: 0x290a130
//
__int64 __fastcall sub_290A130(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int64 a5)
{
  __int64 *v7; // r8
  __int64 *v8; // r13
  __int64 *v9; // r14
  __int64 v10; // r9
  int v11; // r11d
  __int64 *v12; // rcx
  unsigned int v13; // edx
  __int64 *v14; // rax
  __int64 v15; // rdi
  __int64 v16; // r12
  unsigned int v17; // esi
  __int64 v18; // r9
  __int64 v19; // rdi
  int v20; // edx
  int v21; // eax
  unsigned __int64 v22; // rax
  unsigned __int64 v23; // rcx
  int v24; // edx
  __int64 *v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // r8
  __int64 v29; // r9
  __int64 v30; // rax
  __int64 *v32; // rax
  void *v33; // rax
  __int64 v34; // rdx
  void *v35; // rsi
  __int64 *v36; // [rsp+18h] [rbp-78h] BYREF
  __int64 v37; // [rsp+20h] [rbp-70h] BYREF
  int v38; // [rsp+28h] [rbp-68h]
  __int64 v39; // [rsp+30h] [rbp-60h] BYREF
  void *src; // [rsp+38h] [rbp-58h]
  __int64 v41; // [rsp+40h] [rbp-50h]
  __int64 v42; // [rsp+48h] [rbp-48h]
  __int64 *v43; // [rsp+50h] [rbp-40h] BYREF
  __int64 v44; // [rsp+58h] [rbp-38h]
  _BYTE v45[48]; // [rsp+60h] [rbp-30h] BYREF

  v43 = (__int64 *)v45;
  v39 = 0;
  src = 0;
  v41 = 0;
  v42 = 0;
  v44 = 0;
  sub_2909D90(a2, a1, (__int64)&v39, a5);
  v8 = v43;
  v9 = &v43[(unsigned int)v44];
  if ( v9 != v43 )
  {
    while ( 1 )
    {
      v16 = *v8;
      v17 = *(_DWORD *)(a4 + 24);
      v38 = 0;
      v37 = v16;
      if ( !v17 )
        break;
      v10 = *(_QWORD *)(a4 + 8);
      v11 = 1;
      v12 = 0;
      v13 = (v17 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
      v14 = (__int64 *)(v10 + 16LL * v13);
      v15 = *v14;
      if ( v16 == *v14 )
      {
LABEL_4:
        if ( v9 == ++v8 )
          goto LABEL_24;
      }
      else
      {
        while ( v15 != -4096 )
        {
          if ( v15 != -8192 || v12 )
            v14 = v12;
          v13 = (v17 - 1) & (v11 + v13);
          v7 = (__int64 *)(v10 + 16LL * v13);
          v15 = *v7;
          if ( v16 == *v7 )
            goto LABEL_4;
          ++v11;
          v12 = v14;
          v14 = (__int64 *)(v10 + 16LL * v13);
        }
        if ( !v12 )
          v12 = v14;
        v21 = *(_DWORD *)(a4 + 16);
        ++*(_QWORD *)a4;
        v20 = v21 + 1;
        v36 = v12;
        if ( 4 * (v21 + 1) >= 3 * v17 )
          goto LABEL_7;
        v19 = v16;
        v18 = v17 >> 3;
        if ( v17 - *(_DWORD *)(a4 + 20) - v20 <= (unsigned int)v18 )
          goto LABEL_8;
LABEL_18:
        *(_DWORD *)(a4 + 16) = v20;
        if ( *v12 != -4096 )
          --*(_DWORD *)(a4 + 20);
        *v12 = v19;
        *((_DWORD *)v12 + 2) = v38;
        *((_DWORD *)v12 + 2) = *(_DWORD *)(a4 + 40);
        v22 = *(unsigned int *)(a4 + 40);
        v23 = *(unsigned int *)(a4 + 44);
        v24 = *(_DWORD *)(a4 + 40);
        if ( v22 >= v23 )
        {
          if ( v23 < v22 + 1 )
          {
            sub_C8D5F0(a4 + 32, (const void *)(a4 + 48), v22 + 1, 0x10u, (__int64)v7, v18);
            v22 = *(unsigned int *)(a4 + 40);
          }
          v32 = (__int64 *)(*(_QWORD *)(a4 + 32) + 16 * v22);
          *v32 = v16;
          v32[1] = v16;
          ++*(_DWORD *)(a4 + 40);
          goto LABEL_4;
        }
        v25 = (__int64 *)(*(_QWORD *)(a4 + 32) + 16 * v22);
        if ( v25 )
        {
          *v25 = v16;
          v25[1] = v16;
          v24 = *(_DWORD *)(a4 + 40);
        }
        ++v8;
        *(_DWORD *)(a4 + 40) = v24 + 1;
        if ( v9 == v8 )
          goto LABEL_24;
      }
    }
    ++*(_QWORD *)a4;
    v36 = 0;
LABEL_7:
    v17 *= 2;
LABEL_8:
    sub_D39D40(a4, v17);
    sub_22B1A50(a4, &v37, &v36);
    v19 = v37;
    v12 = v36;
    v20 = *(_DWORD *)(a4 + 16) + 1;
    goto LABEL_18;
  }
LABEL_24:
  sub_C7D6A0(*(_QWORD *)(a3 + 8), 8LL * *(unsigned int *)(a3 + 24), 8);
  v30 = (unsigned int)v42;
  *(_DWORD *)(a3 + 24) = v42;
  if ( (_DWORD)v30 )
  {
    v33 = (void *)sub_C7D670(8 * v30, 8);
    v34 = *(unsigned int *)(a3 + 24);
    v35 = src;
    *(_QWORD *)(a3 + 8) = v33;
    *(_QWORD *)(a3 + 16) = v41;
    memcpy(v33, v35, 8 * v34);
  }
  else
  {
    *(_QWORD *)(a3 + 8) = 0;
    *(_QWORD *)(a3 + 16) = 0;
  }
  sub_28FEFA0(a3 + 32, (__int64)&v43, v26, v27, v28, v29);
  if ( v43 != (__int64 *)v45 )
    _libc_free((unsigned __int64)v43);
  return sub_C7D6A0((__int64)src, 8LL * (unsigned int)v42, 8);
}
