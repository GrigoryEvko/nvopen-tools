// Function: sub_142FB50
// Address: 0x142fb50
//
_QWORD *__fastcall sub_142FB50(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rdx
  unsigned __int8 v5; // r14
  char v6; // r13
  __int64 v7; // rax
  __int64 v8; // r15
  char v9; // r13
  char v10; // r13
  char v11; // r14
  _QWORD *v12; // r14
  __int64 v13; // r8
  _QWORD *result; // rax
  _QWORD *v15; // rbx
  __int64 v16; // rbx
  unsigned int v17; // esi
  __int64 v18; // rdi
  unsigned int v19; // r11d
  _QWORD *v20; // rax
  __int64 v21; // rdx
  int v22; // r14d
  _QWORD *v23; // r9
  int v24; // eax
  int v25; // r10d
  int v26; // r10d
  __int64 v27; // r11
  unsigned int v28; // edx
  __int64 v29; // r8
  int v30; // edi
  _QWORD *v31; // rsi
  int v32; // edi
  int v33; // edi
  __int64 v34; // r8
  _QWORD *v35; // rdx
  int v36; // esi
  unsigned int v37; // ecx
  __int64 v38; // r10
  __int64 v39; // [rsp+8h] [rbp-128h]
  __int64 v41; // [rsp+18h] [rbp-118h]
  bool v43; // [rsp+28h] [rbp-108h]
  __int64 v44; // [rsp+28h] [rbp-108h]
  unsigned int v45; // [rsp+28h] [rbp-108h]
  _QWORD v46[2]; // [rsp+30h] [rbp-100h] BYREF
  _QWORD *v47; // [rsp+40h] [rbp-F0h] BYREF
  __int64 v48; // [rsp+48h] [rbp-E8h]
  _QWORD v49[2]; // [rsp+50h] [rbp-E0h] BYREF
  _QWORD v50[26]; // [rsp+60h] [rbp-D0h] BYREF

  sub_15E64D0(a2);
  if ( v4 )
  {
    v5 = *(_BYTE *)(a2 + 32) & 0xF;
    v43 = (unsigned int)v5 - 7 <= 1;
  }
  else
  {
    v43 = 0;
    v5 = *(_BYTE *)(a2 + 32) & 0xF;
  }
  v6 = *(_BYTE *)(a2 + 33);
  v7 = sub_22077B0(80);
  v8 = v7;
  v9 = (v6 & 0x40) != 0;
  if ( v7 )
  {
    *(_DWORD *)(v7 + 8) = 0;
    *(_QWORD *)(v7 + 16) = 0;
    *(_QWORD *)(v7 + 24) = 0;
    *(_QWORD *)(v7 + 32) = 0;
    v10 = (16 * v43) | v5 | (v9 << 6);
    v11 = *(_BYTE *)(v7 + 12);
    *(_QWORD *)(v7 + 40) = 0;
    *(_QWORD *)(v7 + 48) = 0;
    *(_QWORD *)(v7 + 56) = 0;
    *(_QWORD *)v7 = &unk_49EB498;
    *(_BYTE *)(v7 + 12) = v11 & 0x80 | v10;
    *(_QWORD *)(v7 + 64) = 0;
    *(_QWORD *)(v7 + 72) = 0;
  }
  sub_164A820(*(_QWORD *)(a2 - 24));
  sub_15E4EB0(&v47);
  v12 = v47;
  v41 = v48;
  sub_16C1840(v50);
  sub_16C1A90(v50, v12, v41);
  sub_16C1AA0(v50, v46);
  v13 = v46[0];
  if ( v47 != v49 )
  {
    v39 = v46[0];
    j_j___libc_free_0(v47, v49[0] + 1LL);
    v13 = v39;
  }
  *(_QWORD *)(v8 + 64) = sub_16342D0(a1, v13, 1);
  if ( v43 )
  {
    sub_15E4EB0(&v47);
    v15 = v47;
    v44 = v48;
    sub_16C1840(v50);
    sub_16C1A90(v50, v15, v44);
    sub_16C1AA0(v50, v46);
    v16 = v46[0];
    if ( v47 != v49 )
      j_j___libc_free_0(v47, v49[0] + 1LL);
    v17 = *(_DWORD *)(a3 + 24);
    if ( v17 )
    {
      v18 = *(_QWORD *)(a3 + 8);
      v19 = (v17 - 1) & (37 * v16);
      v20 = (_QWORD *)(v18 + 8LL * v19);
      v21 = *v20;
      if ( v16 == *v20 )
        goto LABEL_8;
      v22 = 1;
      v23 = 0;
      while ( v21 != -1 )
      {
        if ( v21 != -2 || v23 )
          v20 = v23;
        v19 = (v17 - 1) & (v22 + v19);
        v21 = *(_QWORD *)(v18 + 8LL * v19);
        if ( v16 == v21 )
          goto LABEL_8;
        ++v22;
        v23 = v20;
        v20 = (_QWORD *)(v18 + 8LL * v19);
      }
      if ( !v23 )
        v23 = v20;
      ++*(_QWORD *)a3;
      v24 = *(_DWORD *)(a3 + 16) + 1;
      if ( 4 * v24 < 3 * v17 )
      {
        if ( v17 - *(_DWORD *)(a3 + 20) - v24 > v17 >> 3 )
        {
LABEL_22:
          *(_DWORD *)(a3 + 16) = v24;
          if ( *v23 != -1 )
            --*(_DWORD *)(a3 + 20);
          *v23 = v16;
          goto LABEL_8;
        }
        v45 = 37 * v16;
        sub_142F750(a3, v17);
        v32 = *(_DWORD *)(a3 + 24);
        if ( v32 )
        {
          v33 = v32 - 1;
          v34 = *(_QWORD *)(a3 + 8);
          v35 = 0;
          v36 = 1;
          v37 = v33 & v45;
          v23 = (_QWORD *)(v34 + 8LL * (v33 & v45));
          v38 = *v23;
          v24 = *(_DWORD *)(a3 + 16) + 1;
          if ( v16 != *v23 )
          {
            while ( v38 != -1 )
            {
              if ( !v35 && v38 == -2 )
                v35 = v23;
              v37 = v33 & (v36 + v37);
              v23 = (_QWORD *)(v34 + 8LL * v37);
              v38 = *v23;
              if ( v16 == *v23 )
                goto LABEL_22;
              ++v36;
            }
            if ( v35 )
              v23 = v35;
          }
          goto LABEL_22;
        }
LABEL_54:
        ++*(_DWORD *)(a3 + 16);
        BUG();
      }
    }
    else
    {
      ++*(_QWORD *)a3;
    }
    sub_142F750(a3, 2 * v17);
    v25 = *(_DWORD *)(a3 + 24);
    if ( v25 )
    {
      v26 = v25 - 1;
      v27 = *(_QWORD *)(a3 + 8);
      v28 = v26 & (37 * v16);
      v23 = (_QWORD *)(v27 + 8LL * v28);
      v29 = *v23;
      v24 = *(_DWORD *)(a3 + 16) + 1;
      if ( v16 != *v23 )
      {
        v30 = 1;
        v31 = 0;
        while ( v29 != -1 )
        {
          if ( !v31 && v29 == -2 )
            v31 = v23;
          v28 = v26 & (v30 + v28);
          v23 = (_QWORD *)(v27 + 8LL * v28);
          v29 = *v23;
          if ( v16 == *v23 )
            goto LABEL_22;
          ++v30;
        }
        if ( v31 )
          v23 = v31;
      }
      goto LABEL_22;
    }
    goto LABEL_54;
  }
LABEL_8:
  v50[0] = v8;
  result = sub_142ED30(a1, a2, v50);
  if ( v50[0] )
    return (_QWORD *)(*(__int64 (__fastcall **)(_QWORD))(*(_QWORD *)v50[0] + 8LL))(v50[0]);
  return result;
}
