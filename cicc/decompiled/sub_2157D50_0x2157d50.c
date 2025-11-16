// Function: sub_2157D50
// Address: 0x2157d50
//
__int64 __fastcall sub_2157D50(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  int v8; // edx
  int v9; // ecx
  __int64 v10; // rdi
  int v11; // r8d
  unsigned int v12; // edx
  __int64 v13; // rsi
  unsigned int v14; // esi
  __int64 v15; // r9
  int v16; // r11d
  unsigned int v17; // edx
  __int64 *v18; // rdi
  __int64 *v19; // rcx
  __int64 v20; // r8
  int v21; // edi
  int v22; // edi
  int v23; // r8d
  __int64 v24; // rax
  __int64 v25; // rbx
  __int64 v26; // rdi
  _QWORD *v27; // rbx
  _QWORD *v28; // r15
  __int64 v29; // rax
  unsigned int v30; // esi
  __int64 v31; // rdx
  __int64 v32; // r8
  unsigned int v33; // eax
  __int64 *v34; // rcx
  __int64 v35; // rdi
  int v36; // eax
  int v37; // edx
  __int64 v38; // r8
  unsigned int v39; // eax
  __int64 *v40; // rsi
  __int64 v41; // rdi
  int v42; // r11d
  __int64 *v43; // r10
  int v44; // eax
  int v45; // ecx
  int v46; // esi
  int v47; // r9d
  __int64 v48; // [rsp+0h] [rbp-70h]
  __int64 v49[2]; // [rsp+8h] [rbp-68h] BYREF
  __int64 *v50; // [rsp+18h] [rbp-58h] BYREF
  __int64 *v51; // [rsp+20h] [rbp-50h] BYREF
  _QWORD *v52; // [rsp+28h] [rbp-48h]
  __int64 v53; // [rsp+30h] [rbp-40h]
  __int64 v54; // [rsp+38h] [rbp-38h]

  result = a1;
  v8 = *(_DWORD *)(a3 + 24);
  v49[0] = a1;
  if ( v8 )
  {
    v9 = v8 - 1;
    v10 = *(_QWORD *)(a3 + 8);
    v11 = 1;
    v12 = (v8 - 1) & (((unsigned int)result >> 9) ^ ((unsigned int)result >> 4));
    v13 = *(_QWORD *)(v10 + 8LL * v12);
    if ( result == v13 )
      return result;
    while ( v13 != -8 )
    {
      v12 = v9 & (v11 + v12);
      v13 = *(_QWORD *)(v10 + 8LL * v12);
      if ( result == v13 )
        return result;
      ++v11;
    }
  }
  v14 = *(_DWORD *)(a4 + 24);
  if ( v14 )
  {
    v15 = *(_QWORD *)(a4 + 8);
    v16 = 1;
    v17 = (v14 - 1) & (((unsigned int)result >> 9) ^ ((unsigned int)result >> 4));
    v18 = (__int64 *)(v15 + 8LL * v17);
    v19 = 0;
    v20 = *v18;
    if ( result == *v18 )
LABEL_9:
      sub_16BD130("Circular dependency found in global variable set", 1u);
    while ( v20 != -8 )
    {
      if ( !v19 && v20 == -16 )
        v19 = v18;
      v17 = (v14 - 1) & (v16 + v17);
      v18 = (__int64 *)(v15 + 8LL * v17);
      v20 = *v18;
      if ( result == *v18 )
        goto LABEL_9;
      ++v16;
    }
    if ( !v19 )
      v19 = v18;
    v21 = *(_DWORD *)(a4 + 16);
    ++*(_QWORD *)a4;
    v22 = v21 + 1;
    if ( 4 * v22 < 3 * v14 )
    {
      v23 = v14 >> 3;
      if ( v14 - *(_DWORD *)(a4 + 20) - v22 > v14 >> 3 )
        goto LABEL_20;
      goto LABEL_64;
    }
  }
  else
  {
    ++*(_QWORD *)a4;
  }
  v14 *= 2;
LABEL_64:
  sub_21579F0(a4, v14);
  sub_21552D0(a4, v49, &v51);
  v19 = v51;
  result = v49[0];
  v22 = *(_DWORD *)(a4 + 16) + 1;
LABEL_20:
  *(_DWORD *)(a4 + 16) = v22;
  if ( *v19 != -8 )
    --*(_DWORD *)(a4 + 20);
  *v19 = result;
  v24 = v49[0];
  v51 = 0;
  v52 = 0;
  v53 = 0;
  v54 = 0;
  if ( (*(_DWORD *)(v49[0] + 20) & 0xFFFFFFF) != 0 )
  {
    v25 = -24;
    v48 = 24LL * ((*(_DWORD *)(v49[0] + 20) & 0xFFFFFFFu) - 1);
    while ( 1 )
    {
      v26 = *(_QWORD *)(v24 + v25);
      v25 += 24;
      sub_2157BA0(v26, (__int64)&v51);
      if ( v48 == v25 )
        break;
      v24 = v49[0];
    }
    v27 = v52;
    v28 = &v52[(unsigned int)v54];
    if ( (_DWORD)v53 )
    {
      if ( v52 != v28 )
      {
        while ( *v27 == -8 || *v27 == -16 )
        {
          if ( ++v27 == v28 )
            goto LABEL_27;
        }
LABEL_40:
        if ( v27 != v28 )
        {
          sub_2157D50(*v27, a2, a3, a4);
          while ( ++v27 != v28 )
          {
            if ( *v27 != -8 && *v27 != -16 )
              goto LABEL_40;
          }
        }
      }
    }
  }
LABEL_27:
  v29 = *(unsigned int *)(a2 + 8);
  if ( (unsigned int)v29 >= *(_DWORD *)(a2 + 12) )
  {
    sub_16CD150(a2, (const void *)(a2 + 16), 0, 8, v23, v15);
    v29 = *(unsigned int *)(a2 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a2 + 8 * v29) = v49[0];
  ++*(_DWORD *)(a2 + 8);
  v30 = *(_DWORD *)(a3 + 24);
  if ( !v30 )
  {
    ++*(_QWORD *)a3;
    goto LABEL_60;
  }
  v31 = v49[0];
  v32 = *(_QWORD *)(a3 + 8);
  v33 = (v30 - 1) & ((LODWORD(v49[0]) >> 9) ^ (LODWORD(v49[0]) >> 4));
  v34 = (__int64 *)(v32 + 8LL * v33);
  v35 = *v34;
  if ( *v34 != v49[0] )
  {
    v42 = 1;
    v43 = 0;
    while ( v35 != -8 )
    {
      if ( !v43 && v35 == -16 )
        v43 = v34;
      v33 = (v30 - 1) & (v42 + v33);
      v34 = (__int64 *)(v32 + 8LL * v33);
      v35 = *v34;
      if ( v49[0] == *v34 )
        goto LABEL_31;
      ++v42;
    }
    v44 = *(_DWORD *)(a3 + 16);
    if ( !v43 )
      v43 = v34;
    ++*(_QWORD *)a3;
    v45 = v44 + 1;
    if ( 4 * (v44 + 1) < 3 * v30 )
    {
      if ( v30 - *(_DWORD *)(a3 + 20) - v45 > v30 >> 3 )
      {
LABEL_52:
        *(_DWORD *)(a3 + 16) = v45;
        if ( *v43 != -8 )
          --*(_DWORD *)(a3 + 20);
        *v43 = v31;
        goto LABEL_31;
      }
LABEL_61:
      sub_21579F0(a3, v30);
      sub_21552D0(a3, v49, &v50);
      v43 = v50;
      v31 = v49[0];
      v45 = *(_DWORD *)(a3 + 16) + 1;
      goto LABEL_52;
    }
LABEL_60:
    v30 *= 2;
    goto LABEL_61;
  }
LABEL_31:
  v36 = *(_DWORD *)(a4 + 24);
  if ( v36 )
  {
    v37 = v36 - 1;
    v38 = *(_QWORD *)(a4 + 8);
    v39 = (v36 - 1) & ((LODWORD(v49[0]) >> 9) ^ (LODWORD(v49[0]) >> 4));
    v40 = (__int64 *)(v38 + 8LL * v39);
    v41 = *v40;
    if ( v49[0] == *v40 )
    {
LABEL_33:
      *v40 = -16;
      --*(_DWORD *)(a4 + 16);
      ++*(_DWORD *)(a4 + 20);
    }
    else
    {
      v46 = 1;
      while ( v41 != -8 )
      {
        v47 = v46 + 1;
        v39 = v37 & (v46 + v39);
        v40 = (__int64 *)(v38 + 8LL * v39);
        v41 = *v40;
        if ( v49[0] == *v40 )
          goto LABEL_33;
        v46 = v47;
      }
    }
  }
  return j___libc_free_0(v52);
}
