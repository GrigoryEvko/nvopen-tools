// Function: sub_2182980
// Address: 0x2182980
//
__int64 __fastcall sub_2182980(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v4; // rbx
  _QWORD *v5; // rax
  __int64 result; // rax
  _QWORD *v7; // rdi
  __int64 v8; // rbx
  __int64 *v9; // r15
  __int64 *v10; // rbx
  __int64 v11; // r8
  int v12; // r11d
  __int64 *v13; // r10
  unsigned int v14; // edx
  __int64 *v15; // rdi
  __int64 v16; // rcx
  unsigned int v17; // esi
  int v18; // esi
  int v19; // esi
  __int64 v20; // r8
  int v21; // edx
  unsigned int v22; // ecx
  _QWORD *v23; // rax
  __int64 v24; // rdi
  __int64 *v25; // rbx
  __int64 *v26; // r12
  __int64 v27; // rdi
  int v28; // edi
  int v29; // esi
  int v30; // esi
  __int64 v31; // r8
  int v32; // r11d
  __int64 *v33; // r9
  unsigned int v34; // ecx
  __int64 v35; // rdi
  __int64 v36; // rdx
  int v37; // r11d
  __int64 *v38; // r9
  __int64 v39; // [rsp+8h] [rbp-B8h] BYREF
  _QWORD v40[6]; // [rsp+10h] [rbp-B0h] BYREF
  __int64 v41; // [rsp+40h] [rbp-80h] BYREF
  __int64 v42; // [rsp+48h] [rbp-78h]
  _QWORD *v43; // [rsp+50h] [rbp-70h]
  _QWORD *v44; // [rsp+58h] [rbp-68h]
  _QWORD *v45; // [rsp+60h] [rbp-60h]
  unsigned __int64 v46; // [rsp+68h] [rbp-58h]
  _QWORD *v47; // [rsp+70h] [rbp-50h]
  _QWORD *v48; // [rsp+78h] [rbp-48h]
  _QWORD *v49; // [rsp+80h] [rbp-40h]
  _QWORD *v50; // [rsp+88h] [rbp-38h]

  v39 = a2;
  v47 = 0;
  v42 = 8;
  v41 = sub_22077B0(64);
  v4 = (_QWORD *)(v41 + 24);
  v5 = (_QWORD *)sub_22077B0(512);
  v46 = v41 + 24;
  *(_QWORD *)(v41 + 24) = v5;
  v44 = v5;
  v45 = v5 + 64;
  v50 = v4;
  v48 = v5;
  v49 = v5 + 64;
  v43 = v5;
  if ( v5 )
    *v5 = v39;
  v47 = v5 + 1;
  result = sub_2182830((__int64)v40, a3, &v39);
  v7 = v47;
  if ( v43 != v47 )
  {
    while ( 1 )
    {
      if ( v48 == v7 )
      {
        v8 = *(_QWORD *)(*(v50 - 1) + 504LL);
        j_j___libc_free_0(v7, 512);
        v36 = *--v50 + 512LL;
        v48 = (_QWORD *)*v50;
        result = (__int64)(v48 + 63);
        v49 = (_QWORD *)v36;
        v47 = v48 + 63;
      }
      else
      {
        v8 = *(v7 - 1);
        v47 = v7 - 1;
      }
      v9 = *(__int64 **)(v8 + 96);
      v10 = *(__int64 **)(v8 + 88);
      if ( v9 != v10 )
        break;
LABEL_20:
      v7 = v47;
      if ( v43 == v47 )
        goto LABEL_21;
    }
    while ( 1 )
    {
      result = *v10;
      v17 = *(_DWORD *)(a3 + 24);
      v40[0] = *v10;
      if ( !v17 )
        break;
      v11 = *(_QWORD *)(a3 + 8);
      v12 = 1;
      v13 = 0;
      v14 = (v17 - 1) & (((unsigned int)result >> 9) ^ ((unsigned int)result >> 4));
      v15 = (__int64 *)(v11 + 8LL * v14);
      v16 = *v15;
      if ( result == *v15 )
      {
LABEL_9:
        if ( v9 == ++v10 )
          goto LABEL_20;
      }
      else
      {
        while ( v16 != -8 )
        {
          if ( v13 || v16 != -16 )
            v15 = v13;
          v14 = (v17 - 1) & (v12 + v14);
          v16 = *(_QWORD *)(v11 + 8LL * v14);
          if ( result == v16 )
            goto LABEL_9;
          ++v12;
          v13 = v15;
          v15 = (__int64 *)(v11 + 8LL * v14);
        }
        if ( !v13 )
          v13 = v15;
        v28 = *(_DWORD *)(a3 + 16);
        ++*(_QWORD *)a3;
        v21 = v28 + 1;
        if ( 4 * (v28 + 1) >= 3 * v17 )
          goto LABEL_12;
        if ( v17 - *(_DWORD *)(a3 + 20) - v21 <= v17 >> 3 )
        {
          sub_1DF9CE0(a3, v17);
          v29 = *(_DWORD *)(a3 + 24);
          if ( !v29 )
          {
LABEL_60:
            ++*(_DWORD *)(a3 + 16);
            BUG();
          }
          v30 = v29 - 1;
          v31 = *(_QWORD *)(a3 + 8);
          v32 = 1;
          v33 = 0;
          result = v40[0];
          v34 = v30 & ((LODWORD(v40[0]) >> 9) ^ (LODWORD(v40[0]) >> 4));
          v13 = (__int64 *)(v31 + 8LL * v34);
          v35 = *v13;
          v21 = *(_DWORD *)(a3 + 16) + 1;
          if ( *v13 != v40[0] )
          {
            while ( v35 != -8 )
            {
              if ( !v33 && v35 == -16 )
                v33 = v13;
              v34 = v30 & (v32 + v34);
              v13 = (__int64 *)(v31 + 8LL * v34);
              v35 = *v13;
              if ( v40[0] == *v13 )
                goto LABEL_14;
              ++v32;
            }
            if ( v33 )
              v13 = v33;
          }
        }
LABEL_14:
        *(_DWORD *)(a3 + 16) = v21;
        if ( *v13 != -8 )
          --*(_DWORD *)(a3 + 20);
        *v13 = result;
        v23 = v47;
        if ( v47 == v49 - 1 )
        {
          result = sub_217F060(&v41, v40);
          goto LABEL_9;
        }
        if ( v47 )
        {
          *v47 = v40[0];
          v23 = v47;
        }
        result = (__int64)(v23 + 1);
        ++v10;
        v47 = (_QWORD *)result;
        if ( v9 == v10 )
          goto LABEL_20;
      }
    }
    ++*(_QWORD *)a3;
LABEL_12:
    sub_1DF9CE0(a3, 2 * v17);
    v18 = *(_DWORD *)(a3 + 24);
    if ( !v18 )
      goto LABEL_60;
    v19 = v18 - 1;
    v20 = *(_QWORD *)(a3 + 8);
    v21 = *(_DWORD *)(a3 + 16) + 1;
    v22 = v19 & ((LODWORD(v40[0]) >> 9) ^ (LODWORD(v40[0]) >> 4));
    v13 = (__int64 *)(v20 + 8LL * v22);
    result = *v13;
    if ( v40[0] != *v13 )
    {
      v37 = 1;
      v38 = 0;
      while ( result != -8 )
      {
        if ( result == -16 && !v38 )
          v38 = v13;
        v22 = v19 & (v37 + v22);
        v13 = (__int64 *)(v20 + 8LL * v22);
        result = *v13;
        if ( v40[0] == *v13 )
          goto LABEL_14;
        ++v37;
      }
      result = v40[0];
      if ( v38 )
        v13 = v38;
    }
    goto LABEL_14;
  }
LABEL_21:
  v24 = v41;
  if ( v41 )
  {
    v25 = (__int64 *)v46;
    v26 = v50 + 1;
    if ( (unsigned __int64)(v50 + 1) > v46 )
    {
      do
      {
        v27 = *v25++;
        j_j___libc_free_0(v27, 512);
      }
      while ( v26 > v25 );
      v24 = v41;
    }
    return j_j___libc_free_0(v24, 8 * v42);
  }
  return result;
}
