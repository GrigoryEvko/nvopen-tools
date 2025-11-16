// Function: sub_1BE5A10
// Address: 0x1be5a10
//
__int64 __fastcall sub_1BE5A10(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v4; // r8
  __int64 v5; // rdi
  __int64 v6; // r15
  __int64 v7; // r12
  _QWORD *v8; // rax
  __int64 v9; // r13
  __int64 v10; // rax
  __int64 *v11; // r12
  __int64 v12; // r9
  int v13; // esi
  unsigned int v14; // ecx
  __int64 *v15; // rax
  __int64 v16; // r10
  __int64 v17; // r14
  unsigned __int64 v18; // rdi
  __int64 v19; // rax
  __int64 v20; // r15
  char v21; // di
  unsigned int v22; // esi
  _QWORD *v23; // rdi
  unsigned int v25; // eax
  __int64 *v26; // r8
  int v27; // ecx
  unsigned int v28; // r9d
  int v29; // r11d
  int v30; // eax
  __int64 v31; // rcx
  int v32; // esi
  unsigned int v33; // eax
  __int64 v34; // rdi
  int v35; // ecx
  __int64 v36; // rsi
  int v37; // ecx
  unsigned int v38; // eax
  __int64 v39; // rdi
  int v40; // r10d
  __int64 *v41; // r9
  __int64 *v42; // rdx
  int v43; // r10d
  __int64 v44; // [rsp+8h] [rbp-68h]
  __int64 v46; // [rsp+18h] [rbp-58h]
  __int64 *v47; // [rsp+18h] [rbp-58h]
  __int64 v48; // [rsp+20h] [rbp-50h] BYREF
  __int16 v49; // [rsp+30h] [rbp-40h]

  v2 = a1 + 16;
  v4 = *(_QWORD *)(a2 + 16);
  v5 = *(_QWORD *)(a2 + 8);
  v49 = 260;
  v46 = v4;
  v6 = *(_QWORD *)(v5 + 56);
  v48 = v2;
  v7 = sub_157E9C0(v5);
  v8 = (_QWORD *)sub_22077B0(64);
  v9 = (__int64)v8;
  if ( v8 )
    sub_157FB60(v8, v7, (__int64)&v48, v6, v46);
  v10 = sub_1BE23D0(a1);
  v11 = *(__int64 **)(v10 + 56);
  v44 = a2 + 24;
  v47 = &v11[*(unsigned int *)(v10 + 64)];
  if ( v11 != v47 )
  {
    while ( 1 )
    {
      v19 = sub_1BE2380(*v11);
      v20 = v19;
      v21 = *(_BYTE *)(a2 + 32) & 1;
      if ( v21 )
      {
        v12 = a2 + 40;
        v13 = 3;
      }
      else
      {
        v22 = *(_DWORD *)(a2 + 48);
        v12 = *(_QWORD *)(a2 + 40);
        if ( !v22 )
        {
          v25 = *(_DWORD *)(a2 + 32);
          ++*(_QWORD *)(a2 + 24);
          v26 = 0;
          v27 = (v25 >> 1) + 1;
LABEL_18:
          v28 = 3 * v22;
          goto LABEL_19;
        }
        v13 = v22 - 1;
      }
      v14 = v13 & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
      v15 = (__int64 *)(v12 + 16LL * v14);
      v16 = *v15;
      if ( v20 == *v15 )
      {
        v17 = v15[1];
        goto LABEL_8;
      }
      v29 = 1;
      v26 = 0;
      while ( v16 != -8 )
      {
        if ( v26 || v16 != -16 )
          v15 = v26;
        v14 = v13 & (v29 + v14);
        v42 = (__int64 *)(v12 + 16LL * v14);
        v16 = *v42;
        if ( v20 == *v42 )
        {
          v17 = v42[1];
          goto LABEL_8;
        }
        ++v29;
        v26 = v15;
        v15 = (__int64 *)(v12 + 16LL * v14);
      }
      v28 = 12;
      v22 = 4;
      if ( !v26 )
        v26 = v15;
      v25 = *(_DWORD *)(a2 + 32);
      ++*(_QWORD *)(a2 + 24);
      v27 = (v25 >> 1) + 1;
      if ( !v21 )
      {
        v22 = *(_DWORD *)(a2 + 48);
        goto LABEL_18;
      }
LABEL_19:
      if ( 4 * v27 >= v28 )
      {
        sub_1BE5630(v44, 2 * v22);
        if ( (*(_BYTE *)(a2 + 32) & 1) != 0 )
        {
          v31 = a2 + 40;
          v32 = 3;
        }
        else
        {
          v30 = *(_DWORD *)(a2 + 48);
          v31 = *(_QWORD *)(a2 + 40);
          if ( !v30 )
            goto LABEL_63;
          v32 = v30 - 1;
        }
        v33 = v32 & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
        v26 = (__int64 *)(v31 + 16LL * v33);
        v34 = *v26;
        if ( v20 == *v26 )
          goto LABEL_34;
        v43 = 1;
        v41 = 0;
        while ( v34 != -8 )
        {
          if ( !v41 && v34 == -16 )
            v41 = v26;
          v33 = v32 & (v43 + v33);
          v26 = (__int64 *)(v31 + 16LL * v33);
          v34 = *v26;
          if ( v20 == *v26 )
            goto LABEL_34;
          ++v43;
        }
        goto LABEL_41;
      }
      if ( v22 - *(_DWORD *)(a2 + 36) - v27 > v22 >> 3 )
        goto LABEL_21;
      sub_1BE5630(v44, v22);
      if ( (*(_BYTE *)(a2 + 32) & 1) != 0 )
      {
        v36 = a2 + 40;
        v37 = 3;
      }
      else
      {
        v35 = *(_DWORD *)(a2 + 48);
        v36 = *(_QWORD *)(a2 + 40);
        if ( !v35 )
        {
LABEL_63:
          *(_DWORD *)(a2 + 32) = (2 * (*(_DWORD *)(a2 + 32) >> 1) + 2) | *(_DWORD *)(a2 + 32) & 1;
          BUG();
        }
        v37 = v35 - 1;
      }
      v38 = v37 & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
      v26 = (__int64 *)(v36 + 16LL * v38);
      v39 = *v26;
      if ( v20 != *v26 )
      {
        v40 = 1;
        v41 = 0;
        while ( v39 != -8 )
        {
          if ( !v41 && v39 == -16 )
            v41 = v26;
          v38 = v37 & (v40 + v38);
          v26 = (__int64 *)(v36 + 16LL * v38);
          v39 = *v26;
          if ( v20 == *v26 )
            goto LABEL_34;
          ++v40;
        }
LABEL_41:
        if ( v41 )
          v26 = v41;
      }
LABEL_34:
      v25 = *(_DWORD *)(a2 + 32);
LABEL_21:
      *(_DWORD *)(a2 + 32) = (2 * (v25 >> 1) + 2) | v25 & 1;
      if ( *v26 != -8 )
        --*(_DWORD *)(a2 + 36);
      *v26 = v20;
      v17 = 0;
      v26[1] = 0;
LABEL_8:
      v18 = sub_157EBA0(v17);
      if ( *(_BYTE *)(v18 + 16) == 31 )
      {
        sub_15F20C0((_QWORD *)v18);
        v23 = sub_1648A60(56, 1u);
        if ( !v23 )
          goto LABEL_10;
        ++v11;
        sub_15F8590((__int64)v23, v9, v17);
        if ( v47 == v11 )
          return v9;
      }
      else
      {
        sub_15F4ED0(v18, **(_QWORD **)(v20 + 80) != a1, v9);
LABEL_10:
        if ( v47 == ++v11 )
          return v9;
      }
    }
  }
  return v9;
}
