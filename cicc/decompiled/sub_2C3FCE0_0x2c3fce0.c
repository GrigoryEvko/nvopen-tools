// Function: sub_2C3FCE0
// Address: 0x2c3fce0
//
unsigned __int64 __fastcall sub_2C3FCE0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 result; // rax
  unsigned __int64 v5; // rdx
  __int64 *v6; // r12
  __int64 v7; // rax
  unsigned int v8; // r13d
  __int64 *v9; // r14
  __int64 v11; // rdi
  int v12; // r10d
  __int64 v13; // r9
  unsigned int v14; // edx
  __int64 v15; // rbx
  __int64 v16; // rcx
  __int64 v17; // r11
  __int64 v18; // rdx
  unsigned int v19; // esi
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rcx
  int v23; // edi
  __int64 v24; // [rsp-108h] [rbp-108h]
  __int64 v25; // [rsp-100h] [rbp-100h]
  __int64 v26; // [rsp-100h] [rbp-100h]
  __int64 v27; // [rsp-100h] [rbp-100h]
  __int64 v28; // [rsp-F8h] [rbp-F8h]
  __int64 v29; // [rsp-F8h] [rbp-F8h]
  unsigned __int64 v30; // [rsp-F8h] [rbp-F8h]
  __int64 *v31; // [rsp-E8h] [rbp-E8h]
  _QWORD v32[3]; // [rsp-D0h] [rbp-D0h] BYREF
  __int64 v33; // [rsp-B8h] [rbp-B8h] BYREF
  __int64 v34; // [rsp-88h] [rbp-88h] BYREF
  char *v35; // [rsp-80h] [rbp-80h] BYREF
  __int64 v36; // [rsp-78h] [rbp-78h]
  _BYTE v37[112]; // [rsp-70h] [rbp-70h] BYREF

  result = *(_QWORD *)(a2 + 16);
  v5 = result & 0xFFFFFFFFFFFFFFF8LL;
  if ( (result & 0xFFFFFFFFFFFFFFF8LL) != 0 )
  {
    if ( (result & 4) != 0 )
    {
      v6 = *(__int64 **)v5;
      v7 = *(unsigned int *)(v5 + 8);
    }
    else
    {
      v6 = (__int64 *)(a2 + 16);
      v7 = 1;
    }
    result = (unsigned __int64)&v6[v7];
    v31 = (__int64 *)result;
    if ( (__int64 *)result != v6 )
    {
      v24 = a1 + 160;
      v8 = 0;
      v9 = v6;
      while ( 1 )
      {
        v32[2] = 0x600000000LL;
        v19 = *(_DWORD *)(a1 + 184);
        v32[1] = &v33;
        v20 = *v9;
        v35 = v37;
        v34 = v20;
        v36 = 0x600000000LL;
        if ( !v19 )
          break;
        v11 = *(_QWORD *)(a1 + 168);
        v12 = 1;
        v13 = 0;
        v14 = (v19 - 1) & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
        v15 = v11 + 72LL * v14;
        v16 = *(_QWORD *)v15;
        if ( v20 != *(_QWORD *)v15 )
        {
          while ( v16 != -4096 )
          {
            if ( v16 == -8192 && !v13 )
              v13 = v15;
            v14 = (v19 - 1) & (v12 + v14);
            v15 = v11 + 72LL * v14;
            v16 = *(_QWORD *)v15;
            if ( v20 == *(_QWORD *)v15 )
              goto LABEL_7;
            ++v12;
          }
          v23 = *(_DWORD *)(a1 + 176);
          if ( v13 )
            v15 = v13;
          ++*(_QWORD *)(a1 + 160);
          v22 = (unsigned int)(v23 + 1);
          v32[0] = v15;
          if ( 4 * (int)v22 < 3 * v19 )
          {
            v21 = v19 - *(_DWORD *)(a1 + 180) - (unsigned int)v22;
            if ( (unsigned int)v21 <= v19 >> 3 )
            {
              v28 = a3;
LABEL_16:
              sub_2C3F4B0(v24, v19);
              sub_2C3F050(v24, &v34, v32);
              v20 = v34;
              v15 = v32[0];
              a3 = v28;
              v22 = (unsigned int)(*(_DWORD *)(a1 + 176) + 1);
            }
            *(_DWORD *)(a1 + 176) = v22;
            if ( *(_QWORD *)v15 != -4096 )
              --*(_DWORD *)(a1 + 180);
            *(_QWORD *)v15 = v20;
            v17 = v15 + 8;
            *(_QWORD *)(v15 + 8) = v15 + 24;
            *(_QWORD *)(v15 + 16) = 0x600000000LL;
            if ( (_DWORD)v36 )
            {
              v26 = a3;
              sub_2C3D860(v15 + 8, &v35, v21, v22, a3, v13);
              a3 = v26;
              v17 = v15 + 8;
            }
            if ( v35 != v37 )
            {
              v25 = a3;
              v29 = v17;
              _libc_free((unsigned __int64)v35);
              a3 = v25;
              v17 = v29;
            }
            goto LABEL_8;
          }
LABEL_15:
          v28 = a3;
          v19 *= 2;
          goto LABEL_16;
        }
LABEL_7:
        v17 = v15 + 8;
LABEL_8:
        result = *(_QWORD *)(a3 + 16) & 0xFFFFFFFFFFFFFFF8LL;
        if ( (*(_QWORD *)(a3 + 16) & 4) != 0 )
          result = *(_QWORD *)(*(_QWORD *)result + 8LL * v8);
        v18 = *(unsigned int *)(v15 + 16);
        if ( v18 + 1 > (unsigned __int64)*(unsigned int *)(v15 + 20) )
        {
          v27 = a3;
          v30 = result;
          sub_C8D5F0(v17, (const void *)(v15 + 24), v18 + 1, 8u, a3, v15 + 24);
          v18 = *(unsigned int *)(v15 + 16);
          a3 = v27;
          result = v30;
        }
        ++v8;
        ++v9;
        *(_QWORD *)(*(_QWORD *)(v15 + 8) + 8 * v18) = result;
        ++*(_DWORD *)(v15 + 16);
        if ( v31 == v9 )
          return result;
      }
      ++*(_QWORD *)(a1 + 160);
      v32[0] = 0;
      goto LABEL_15;
    }
  }
  return result;
}
