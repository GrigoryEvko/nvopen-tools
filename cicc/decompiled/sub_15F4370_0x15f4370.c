// Function: sub_15F4370
// Address: 0x15f4370
//
void __fastcall sub_15F4370(__int64 a1, __int64 a2, int *a3, __int64 a4)
{
  __int64 v5; // r13
  int *v6; // rbx
  unsigned int v7; // esi
  int *v8; // r13
  _DWORD *v9; // rax
  __int64 *v10; // r11
  __int64 v11; // r9
  unsigned int v12; // edi
  _DWORD *v13; // rcx
  int v14; // edx
  int v15; // r12d
  int v16; // ecx
  _DWORD *v17; // r10
  int v18; // esi
  int v19; // eax
  __int64 v20; // rsi
  __int64 v21; // r13
  __int64 v22; // rsi
  unsigned int *v23; // r15
  unsigned int *v24; // rbx
  int v25; // eax
  __int64 v26; // rsi
  _DWORD *v27; // rdi
  int v28; // r15d
  int v29; // r8d
  int v30; // ecx
  int v31; // edi
  int v32; // esi
  int v33; // edx
  int v34; // r8d
  int v35; // r15d
  _DWORD *v36; // r8
  __int64 v37; // [rsp+8h] [rbp-C8h]
  __int64 v38; // [rsp+8h] [rbp-C8h]
  __int64 *v39; // [rsp+10h] [rbp-C0h]
  int v40; // [rsp+10h] [rbp-C0h]
  __int64 *v41; // [rsp+10h] [rbp-C0h]
  __int64 v43; // [rsp+28h] [rbp-A8h] BYREF
  __int64 v44; // [rsp+30h] [rbp-A0h] BYREF
  _DWORD *v45; // [rsp+38h] [rbp-98h]
  __int64 v46; // [rsp+40h] [rbp-90h]
  __int64 v47; // [rsp+48h] [rbp-88h]
  unsigned int *v48; // [rsp+50h] [rbp-80h] BYREF
  __int64 v49; // [rsp+58h] [rbp-78h]
  _BYTE v50[112]; // [rsp+60h] [rbp-70h] BYREF

  v5 = a4;
  v6 = a3;
  if ( !*(_QWORD *)(a2 + 48) && *(__int16 *)(a2 + 18) >= 0 )
    return;
  v44 = 0;
  v45 = 0;
  v46 = 0;
  v47 = 0;
  if ( &a3[a4] != a3 )
  {
    v7 = 0;
    v8 = &a3[a4];
    v9 = 0;
    v10 = &v44;
    v11 = a4;
    while ( 1 )
    {
      v15 = *v6;
      if ( !v7 )
        break;
      v12 = (v7 - 1) & (37 * v15);
      v13 = &v9[v12];
      v14 = *v13;
      if ( v15 == *v13 )
      {
LABEL_5:
        if ( v8 == ++v6 )
          goto LABEL_14;
        goto LABEL_6;
      }
      v40 = 1;
      v17 = 0;
      while ( v14 != -1 )
      {
        if ( v17 || v14 != -2 )
          v13 = v17;
        v12 = (v7 - 1) & (v40 + v12);
        v14 = v9[v12];
        if ( v15 == v14 )
          goto LABEL_5;
        ++v40;
        v17 = v13;
        v13 = &v9[v12];
      }
      if ( !v17 )
        v17 = v13;
      ++v44;
      v19 = v46 + 1;
      if ( 4 * ((int)v46 + 1) >= 3 * v7 )
        goto LABEL_9;
      if ( v7 - (v19 + HIDWORD(v46)) <= v7 >> 3 )
      {
        v38 = v11;
        v41 = v10;
        sub_136B240((__int64)v10, v7);
        if ( !(_DWORD)v47 )
        {
LABEL_79:
          LODWORD(v46) = v46 + 1;
          BUG();
        }
        v27 = 0;
        v10 = v41;
        v28 = (v47 - 1) & (37 * v15);
        v11 = v38;
        v29 = 1;
        v17 = &v45[v28];
        v30 = *v17;
        v19 = v46 + 1;
        if ( v15 != *v17 )
        {
          while ( v30 != -1 )
          {
            if ( !v27 && v30 == -2 )
              v27 = v17;
            v28 = (v47 - 1) & (v29 + v28);
            v17 = &v45[v28];
            v30 = *v17;
            if ( v15 == *v17 )
              goto LABEL_11;
            ++v29;
          }
          if ( v27 )
            v17 = v27;
        }
      }
LABEL_11:
      LODWORD(v46) = v19;
      if ( *v17 != -1 )
        --HIDWORD(v46);
      ++v6;
      *v17 = v15;
      if ( v8 == v6 )
      {
LABEL_14:
        v5 = v11;
        goto LABEL_15;
      }
LABEL_6:
      v9 = v45;
      v7 = v47;
    }
    ++v44;
LABEL_9:
    v37 = v11;
    v39 = v10;
    sub_136B240((__int64)v10, 2 * v7);
    if ( !(_DWORD)v47 )
      goto LABEL_79;
    v10 = v39;
    v11 = v37;
    v16 = (v47 - 1) & (37 * v15);
    v17 = &v45[v16];
    v18 = *v17;
    v19 = v46 + 1;
    if ( v15 != *v17 )
    {
      v35 = 1;
      v36 = 0;
      while ( v18 != -1 )
      {
        if ( !v36 && v18 == -2 )
          v36 = v17;
        v16 = (v47 - 1) & (v35 + v16);
        v17 = &v45[v16];
        v18 = *v17;
        if ( v15 == *v17 )
          goto LABEL_11;
        ++v35;
      }
      if ( v36 )
        v17 = v36;
    }
    goto LABEL_11;
  }
LABEL_15:
  v49 = 0x400000000LL;
  v48 = (unsigned int *)v50;
  if ( *(__int16 *)(a2 + 18) < 0 )
  {
    sub_161F980(a2, &v48);
    v23 = v48;
    v24 = &v48[4 * (unsigned int)v49];
    if ( v48 != v24 )
    {
      while ( v5 )
      {
        if ( (_DWORD)v47 )
        {
          v25 = (v47 - 1) & (37 * *v23);
          v26 = (unsigned int)v45[v25];
          if ( *v23 != (_DWORD)v26 )
          {
            v34 = 1;
            while ( (_DWORD)v26 != -1 )
            {
              v25 = (v47 - 1) & (v34 + v25);
              v26 = (unsigned int)v45[v25];
              if ( *v23 == (_DWORD)v26 )
                goto LABEL_30;
              ++v34;
            }
            goto LABEL_31;
          }
          goto LABEL_30;
        }
LABEL_31:
        v23 += 4;
        if ( v24 == v23 )
          goto LABEL_16;
      }
      v26 = *v23;
LABEL_30:
      sub_1625C10(a1, v26, *((_QWORD *)v23 + 1));
      goto LABEL_31;
    }
  }
LABEL_16:
  if ( v5 )
  {
    if ( !(_DWORD)v47 )
      goto LABEL_22;
    v31 = 1;
    v32 = 0;
    v33 = *v45;
    if ( *v45 )
    {
      while ( v33 != -1 )
      {
        v32 = (v47 - 1) & (v31 + v32);
        v33 = v45[v32];
        if ( !v33 )
          goto LABEL_17;
        ++v31;
      }
      goto LABEL_22;
    }
  }
LABEL_17:
  v20 = *(_QWORD *)(a2 + 48);
  v43 = v20;
  if ( v20 )
  {
    v21 = a1 + 48;
    sub_1623A60(&v43, v20, 2);
    if ( !*(_QWORD *)(a1 + 48) )
      goto LABEL_20;
    goto LABEL_19;
  }
  v21 = a1 + 48;
  if ( *(_QWORD *)(a1 + 48) )
  {
LABEL_19:
    sub_161E7C0(v21);
LABEL_20:
    v22 = v43;
    *(_QWORD *)(a1 + 48) = v43;
    if ( v22 )
      sub_1623210(&v43, v22, v21);
  }
LABEL_22:
  if ( v48 != (unsigned int *)v50 )
    _libc_free((unsigned __int64)v48);
  j___libc_free_0(v45);
}
