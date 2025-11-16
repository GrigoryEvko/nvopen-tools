// Function: sub_FF8F20
// Address: 0xff8f20
//
__int64 __fastcall sub_FF8F20(__int64 a1, __int64 a2)
{
  __int64 v3; // rsi
  unsigned int v4; // r15d
  __int64 *v5; // rbx
  unsigned int v6; // r14d
  __int64 v7; // r9
  int v8; // ecx
  __int64 *v9; // rdx
  unsigned int v10; // r8d
  _QWORD *v11; // rax
  __int64 v12; // rdi
  unsigned int *v13; // rax
  unsigned int v14; // esi
  __int64 v15; // r13
  int v16; // eax
  int v17; // esi
  __int64 v18; // r8
  unsigned int v19; // eax
  int v20; // edi
  __int64 v21; // r9
  int v22; // eax
  int v23; // eax
  int v24; // eax
  __int64 v25; // r8
  __int64 *v26; // r9
  unsigned int v27; // r15d
  int v28; // r10d
  __int64 v29; // rsi
  int v31; // r11d
  __int64 *v32; // r10
  __int64 *v33; // [rsp+8h] [rbp-A8h]
  _QWORD v34[2]; // [rsp+10h] [rbp-A0h] BYREF
  __int64 v35; // [rsp+20h] [rbp-90h]
  __int64 v36; // [rsp+28h] [rbp-88h]
  __int64 v37; // [rsp+30h] [rbp-80h]
  __int64 v38; // [rsp+38h] [rbp-78h]
  __int64 v39; // [rsp+40h] [rbp-70h]
  __int64 v40; // [rsp+48h] [rbp-68h]
  __int64 *v41; // [rsp+50h] [rbp-60h]
  __int64 *v42; // [rsp+58h] [rbp-58h]
  __int64 v43; // [rsp+60h] [rbp-50h]
  __int64 v44; // [rsp+68h] [rbp-48h]
  __int64 v45; // [rsp+70h] [rbp-40h]
  __int64 v46; // [rsp+78h] [rbp-38h]

  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  v3 = *(_QWORD *)(a2 + 80);
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  if ( v3 )
    v3 -= 24;
  v4 = 0;
  v34[0] = 0;
  v34[1] = 0;
  v35 = 0;
  v36 = 0;
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
  sub_FF8520((__int64)v34, v3);
  sub_FF89C0((__int64)v34);
  v5 = v41;
  v33 = v42;
  if ( v41 != v42 )
  {
    while ( 1 )
    {
      v6 = v4;
      if ( (char *)v33 - (char *)v5 != 8 )
        break;
LABEL_33:
      ++v4;
      sub_FF89C0((__int64)v34);
      v5 = v41;
      v33 = v42;
      if ( v42 == v41 )
        goto LABEL_34;
    }
    while ( 1 )
    {
      v14 = *(_DWORD *)(a1 + 24);
      v15 = *v5;
      if ( !v14 )
        break;
      v7 = *(_QWORD *)(a1 + 8);
      v8 = 1;
      v9 = 0;
      v10 = (v14 - 1) & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
      v11 = (_QWORD *)(v7 + 16LL * v10);
      v12 = *v11;
      if ( v15 != *v11 )
      {
        while ( v12 != -4096 )
        {
          if ( !v9 && v12 == -8192 )
            v9 = v11;
          v10 = (v14 - 1) & (v8 + v10);
          v11 = (_QWORD *)(v7 + 16LL * v10);
          v12 = *v11;
          if ( v15 == *v11 )
            goto LABEL_7;
          ++v8;
        }
        if ( !v9 )
          v9 = v11;
        v22 = *(_DWORD *)(a1 + 16);
        ++*(_QWORD *)a1;
        v20 = v22 + 1;
        if ( 4 * (v22 + 1) < 3 * v14 )
        {
          if ( v14 - *(_DWORD *)(a1 + 20) - v20 <= v14 >> 3 )
          {
            sub_FF1B10(a1, v14);
            v23 = *(_DWORD *)(a1 + 24);
            if ( !v23 )
            {
LABEL_56:
              ++*(_DWORD *)(a1 + 16);
              BUG();
            }
            v24 = v23 - 1;
            v25 = *(_QWORD *)(a1 + 8);
            v26 = 0;
            v27 = v24 & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
            v28 = 1;
            v20 = *(_DWORD *)(a1 + 16) + 1;
            v9 = (__int64 *)(v25 + 16LL * v27);
            v29 = *v9;
            if ( v15 != *v9 )
            {
              while ( v29 != -4096 )
              {
                if ( !v26 && v29 == -8192 )
                  v26 = v9;
                v27 = v24 & (v28 + v27);
                v9 = (__int64 *)(v25 + 16LL * v27);
                v29 = *v9;
                if ( v15 == *v9 )
                  goto LABEL_13;
                ++v28;
              }
              if ( v26 )
                v9 = v26;
            }
          }
          goto LABEL_13;
        }
LABEL_11:
        sub_FF1B10(a1, 2 * v14);
        v16 = *(_DWORD *)(a1 + 24);
        if ( !v16 )
          goto LABEL_56;
        v17 = v16 - 1;
        v18 = *(_QWORD *)(a1 + 8);
        v19 = (v16 - 1) & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
        v20 = *(_DWORD *)(a1 + 16) + 1;
        v9 = (__int64 *)(v18 + 16LL * v19);
        v21 = *v9;
        if ( v15 != *v9 )
        {
          v31 = 1;
          v32 = 0;
          while ( v21 != -4096 )
          {
            if ( v21 == -8192 && !v32 )
              v32 = v9;
            v19 = v17 & (v31 + v19);
            v9 = (__int64 *)(v18 + 16LL * v19);
            v21 = *v9;
            if ( v15 == *v9 )
              goto LABEL_13;
            ++v31;
          }
          if ( v32 )
            v9 = v32;
        }
LABEL_13:
        *(_DWORD *)(a1 + 16) = v20;
        if ( *v9 != -4096 )
          --*(_DWORD *)(a1 + 20);
        *v9 = v15;
        v13 = (unsigned int *)(v9 + 1);
        *((_DWORD *)v9 + 2) = 0;
        goto LABEL_8;
      }
LABEL_7:
      v13 = (unsigned int *)(v11 + 1);
LABEL_8:
      *v13 = v6;
      ++v5;
      sub_FF1FB0(a1, v15, v6);
      if ( v33 == v5 )
      {
        v4 = v6;
        goto LABEL_33;
      }
    }
    ++*(_QWORD *)a1;
    goto LABEL_11;
  }
LABEL_34:
  if ( v44 )
    j_j___libc_free_0(v44, v46 - v44);
  if ( v41 )
    j_j___libc_free_0(v41, v43 - (_QWORD)v41);
  if ( v38 )
    j_j___libc_free_0(v38, v40 - v38);
  return sub_C7D6A0(v35, 16LL * (unsigned int)v37, 8);
}
