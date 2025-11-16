// Function: sub_13C8A90
// Address: 0x13c8a90
//
__int64 __fastcall sub_13C8A90(__int64 a1, __int64 a2)
{
  __int64 v3; // rsi
  __int64 **v4; // r15
  __int64 **v5; // rbx
  int v6; // edx
  __int64 v7; // r8
  unsigned int v8; // edi
  __int64 *v9; // rax
  __int64 v10; // rcx
  __int64 v11; // r12
  unsigned int v12; // esi
  int v13; // eax
  int v14; // r8d
  __int64 v15; // rsi
  unsigned int v16; // ecx
  int v17; // edi
  __int64 v18; // r9
  int v20; // r11d
  __int64 *v21; // r10
  int v22; // ecx
  int v23; // eax
  int v24; // ecx
  __int64 v25; // r8
  __int64 *v26; // r9
  unsigned int v27; // r14d
  int v28; // r10d
  __int64 v29; // rsi
  int v30; // r11d
  __int64 *v31; // r10
  __int64 v32; // [rsp+8h] [rbp-B8h]
  int v33; // [rsp+1Ch] [rbp-A4h]
  int v34; // [rsp+1Ch] [rbp-A4h]
  int v35; // [rsp+1Ch] [rbp-A4h]
  _QWORD v36[2]; // [rsp+20h] [rbp-A0h] BYREF
  __int64 v37; // [rsp+30h] [rbp-90h]
  __int64 v38; // [rsp+38h] [rbp-88h]
  __int64 v39; // [rsp+40h] [rbp-80h]
  __int64 v40; // [rsp+48h] [rbp-78h]
  __int64 v41; // [rsp+50h] [rbp-70h]
  __int64 v42; // [rsp+58h] [rbp-68h]
  __int64 **v43; // [rsp+60h] [rbp-60h]
  __int64 **v44; // [rsp+68h] [rbp-58h]
  __int64 v45; // [rsp+70h] [rbp-50h]
  __int64 v46; // [rsp+78h] [rbp-48h]
  __int64 v47; // [rsp+80h] [rbp-40h]
  __int64 v48; // [rsp+88h] [rbp-38h]

  v3 = *(_QWORD *)(a2 + 56);
  v36[0] = 0;
  v36[1] = 0;
  v37 = 0;
  v38 = 0;
  v39 = 0;
  v40 = 0;
  v41 = 0;
  v42 = 0;
  v43 = 0;
  v44 = 0;
  v45 = 0;
  v46 = 0;
  v47 = 0;
  v48 = 0;
  sub_13C69A0((__int64)v36, v3);
  sub_13C6E30((__int64)v36);
  v4 = v44;
  v5 = v43;
  v6 = 0;
  v32 = a1 + 296;
  if ( v43 != v44 )
  {
    while ( 1 )
    {
      v11 = **v5;
      if ( v11 )
        break;
LABEL_5:
      if ( v4 == ++v5 )
      {
        v34 = v6 + 1;
        sub_13C6E30((__int64)v36);
        v4 = v44;
        v5 = v43;
        v6 = v34;
        if ( v43 == v44 )
          goto LABEL_15;
      }
    }
    v12 = *(_DWORD *)(a1 + 320);
    if ( v12 )
    {
      v7 = *(_QWORD *)(a1 + 304);
      v8 = (v12 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
      v9 = (__int64 *)(v7 + 16LL * v8);
      v10 = *v9;
      if ( v11 == *v9 )
      {
LABEL_4:
        *((_DWORD *)v9 + 2) = v6;
        goto LABEL_5;
      }
      v20 = 1;
      v21 = 0;
      while ( v10 != -8 )
      {
        if ( v10 == -16 && !v21 )
          v21 = v9;
        v8 = (v12 - 1) & (v20 + v8);
        v9 = (__int64 *)(v7 + 16LL * v8);
        v10 = *v9;
        if ( v11 == *v9 )
          goto LABEL_4;
        ++v20;
      }
      v22 = *(_DWORD *)(a1 + 312);
      if ( v21 )
        v9 = v21;
      ++*(_QWORD *)(a1 + 296);
      v17 = v22 + 1;
      if ( 4 * (v22 + 1) < 3 * v12 )
      {
        if ( v12 - *(_DWORD *)(a1 + 316) - v17 <= v12 >> 3 )
        {
          v35 = v6;
          sub_13C6490(v32, v12);
          v23 = *(_DWORD *)(a1 + 320);
          if ( !v23 )
          {
LABEL_54:
            ++*(_DWORD *)(a1 + 312);
            BUG();
          }
          v24 = v23 - 1;
          v25 = *(_QWORD *)(a1 + 304);
          v26 = 0;
          v27 = (v23 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
          v28 = 1;
          v17 = *(_DWORD *)(a1 + 312) + 1;
          v6 = v35;
          v9 = (__int64 *)(v25 + 16LL * v27);
          v29 = *v9;
          if ( v11 != *v9 )
          {
            while ( v29 != -8 )
            {
              if ( !v26 && v29 == -16 )
                v26 = v9;
              v27 = v24 & (v28 + v27);
              v9 = (__int64 *)(v25 + 16LL * v27);
              v29 = *v9;
              if ( v11 == *v9 )
                goto LABEL_11;
              ++v28;
            }
            if ( v26 )
              v9 = v26;
          }
        }
        goto LABEL_11;
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 296);
    }
    v33 = v6;
    sub_13C6490(v32, 2 * v12);
    v13 = *(_DWORD *)(a1 + 320);
    if ( !v13 )
      goto LABEL_54;
    v14 = v13 - 1;
    v15 = *(_QWORD *)(a1 + 304);
    v16 = (v13 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
    v17 = *(_DWORD *)(a1 + 312) + 1;
    v6 = v33;
    v9 = (__int64 *)(v15 + 16LL * v16);
    v18 = *v9;
    if ( v11 != *v9 )
    {
      v30 = 1;
      v31 = 0;
      while ( v18 != -8 )
      {
        if ( v18 == -16 && !v31 )
          v31 = v9;
        v16 = v14 & (v30 + v16);
        v9 = (__int64 *)(v15 + 16LL * v16);
        v18 = *v9;
        if ( v11 == *v9 )
          goto LABEL_11;
        ++v30;
      }
      if ( v31 )
        v9 = v31;
    }
LABEL_11:
    *(_DWORD *)(a1 + 312) = v17;
    if ( *v9 != -8 )
      --*(_DWORD *)(a1 + 316);
    *v9 = v11;
    *((_DWORD *)v9 + 2) = 0;
    goto LABEL_4;
  }
LABEL_15:
  if ( v46 )
    j_j___libc_free_0(v46, v48 - v46);
  if ( v43 )
    j_j___libc_free_0(v43, v45 - (_QWORD)v43);
  if ( v40 )
    j_j___libc_free_0(v40, v42 - v40);
  return j___libc_free_0(v37);
}
