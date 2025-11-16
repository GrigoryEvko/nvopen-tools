// Function: sub_B1B7B0
// Address: 0xb1b7b0
//
void __fastcall sub_B1B7B0(__int64 a1, __int64 a2)
{
  unsigned __int64 v3; // rax
  int v4; // edx
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rax
  __int64 v7; // r15
  __int64 v8; // rbx
  _BYTE *v9; // rcx
  __int64 v10; // rsi
  __int64 v11; // rax
  __int64 v12; // r14
  _QWORD *v13; // rsi
  _BYTE *v14; // rdx
  __int64 v15; // rsi
  __int64 v16; // r14
  __int64 v17; // rdx
  __int64 v18; // r13
  __int64 v19; // rcx
  unsigned int v20; // edx
  char v21; // bl
  unsigned int v22; // r9d
  __int64 v23; // rax
  __int64 v24; // r8
  unsigned int v25; // edx
  __int64 v26; // r14
  __int64 v27; // rdi
  int v28; // ecx
  __int64 v29; // rax
  _BYTE *v30; // r13
  unsigned int v31; // ebx
  __int64 v32; // r14
  __int64 v33; // r15
  __int64 v34; // r12
  __int64 v35; // rsi
  __int64 v36; // rdi
  unsigned int v37; // eax
  __int64 v38; // rdx
  __int64 v39; // rcx
  unsigned int v40; // eax
  __int64 v41; // rdx
  _QWORD *v42; // rdi
  __int64 v43; // rax
  __int64 v44; // [rsp+0h] [rbp-90h]
  char v45; // [rsp+Fh] [rbp-81h]
  __int64 v46; // [rsp+10h] [rbp-80h]
  __int64 v47; // [rsp+18h] [rbp-78h]
  _BYTE *v49; // [rsp+30h] [rbp-60h] BYREF
  __int64 v50; // [rsp+38h] [rbp-58h]
  _BYTE v51[80]; // [rsp+40h] [rbp-50h] BYREF

  v3 = *(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v3 == a2 + 48 )
  {
    v5 = 0;
  }
  else
  {
    if ( !v3 )
      BUG();
    v4 = *(unsigned __int8 *)(v3 - 24);
    v5 = 0;
    v6 = v3 - 24;
    if ( (unsigned int)(v4 - 30) < 0xB )
      v5 = v6;
  }
  v7 = sub_B46EC0(v5, 0);
  v8 = *(_QWORD *)(a2 + 16);
  if ( v8 )
  {
    while ( 1 )
    {
      v9 = *(_BYTE **)(v8 + 24);
      if ( (unsigned __int8)(*v9 - 30) <= 0xAu )
        break;
      v8 = *(_QWORD *)(v8 + 8);
      if ( !v8 )
        goto LABEL_42;
    }
    v10 = 0;
    v49 = v51;
    v50 = 0x400000000LL;
    v11 = v8;
    while ( 1 )
    {
      v11 = *(_QWORD *)(v11 + 8);
      if ( !v11 )
        break;
      while ( (unsigned __int8)(**(_BYTE **)(v11 + 24) - 30) <= 0xAu )
      {
        v11 = *(_QWORD *)(v11 + 8);
        ++v10;
        if ( !v11 )
          goto LABEL_11;
      }
    }
LABEL_11:
    v12 = v10 + 1;
    if ( v10 + 1 > 4 )
    {
      sub_C8D5F0(&v49, v51, v12, 8);
      v9 = *(_BYTE **)(v8 + 24);
      v13 = &v49[8 * (unsigned int)v50];
    }
    else
    {
      v13 = v51;
    }
    v14 = v9;
LABEL_16:
    if ( v13 )
      *v13 = *((_QWORD *)v14 + 5);
    while ( 1 )
    {
      v8 = *(_QWORD *)(v8 + 8);
      if ( !v8 )
        break;
      v14 = *(_BYTE **)(v8 + 24);
      if ( (unsigned __int8)(*v14 - 30) <= 0xAu )
      {
        ++v13;
        goto LABEL_16;
      }
    }
    v15 = (unsigned int)(v12 + v50);
    v16 = *(_QWORD *)(v7 + 16);
    LODWORD(v50) = v15;
    if ( v16 )
      goto LABEL_20;
LABEL_40:
    v21 = 1;
  }
  else
  {
LABEL_42:
    v16 = *(_QWORD *)(v7 + 16);
    v15 = 0;
    v49 = v51;
    v50 = 0x400000000LL;
    if ( !v16 )
      return;
LABEL_20:
    while ( 1 )
    {
      v17 = *(_QWORD *)(v16 + 24);
      if ( (unsigned __int8)(*(_BYTE *)v17 - 30) <= 0xAu )
        break;
      v16 = *(_QWORD *)(v16 + 8);
      if ( !v16 )
        goto LABEL_40;
    }
LABEL_23:
    v18 = *(_QWORD *)(v17 + 40);
    if ( v18 == a2
      || (unsigned __int8)sub_B19720(a1, v7, *(_QWORD *)(v17 + 40))
      || (!v18 ? (v19 = 0, v20 = 0) : (v19 = (unsigned int)(*(_DWORD *)(v18 + 44) + 1), v20 = *(_DWORD *)(v18 + 44) + 1),
          v20 >= *(_DWORD *)(a1 + 32) || !*(_QWORD *)(*(_QWORD *)(a1 + 24) + 8 * v19)) )
    {
      while ( 1 )
      {
        v16 = *(_QWORD *)(v16 + 8);
        if ( !v16 )
          break;
        v17 = *(_QWORD *)(v16 + 24);
        if ( (unsigned __int8)(*(_BYTE *)v17 - 30) <= 0xAu )
          goto LABEL_23;
      }
      v15 = (unsigned int)v50;
      v21 = 1;
    }
    else
    {
      v15 = (unsigned int)v50;
      v21 = 0;
    }
  }
  if ( !(_DWORD)v15 )
    goto LABEL_69;
  v22 = *(_DWORD *)(a1 + 32);
  v23 = 0;
  while ( 1 )
  {
    v27 = *(_QWORD *)&v49[8 * v23];
    v28 = v23;
    if ( v27 )
    {
      v24 = (unsigned int)(*(_DWORD *)(v27 + 44) + 1);
      v25 = *(_DWORD *)(v27 + 44) + 1;
    }
    else
    {
      v24 = 0;
      v25 = 0;
    }
    if ( v25 < v22 )
    {
      v26 = *(_QWORD *)(a1 + 24);
      if ( *(_QWORD *)(v26 + 8 * v24) )
        break;
    }
    if ( (unsigned int)v15 == ++v23 )
      goto LABEL_69;
  }
  if ( v27 )
  {
    v29 = (unsigned int)(v23 + 1);
    if ( (unsigned int)v29 >= (unsigned int)v15 )
      goto LABEL_57;
    v45 = v21;
    v30 = v49;
    v46 = *(_QWORD *)(a1 + 24);
    v31 = *(_DWORD *)(a1 + 32);
    v44 = v7;
    v32 = a1;
    v33 = (unsigned int)(v28 + 2);
    v47 = v33 + 1 + (unsigned int)(v15 - 2 - v28);
    v34 = v33 + 1;
    v35 = v27;
    while ( 1 )
    {
      v38 = *(_QWORD *)&v30[8 * v29];
      if ( v38 )
      {
        v36 = (unsigned int)(*(_DWORD *)(v38 + 44) + 1);
        v37 = *(_DWORD *)(v38 + 44) + 1;
      }
      else
      {
        v36 = 0;
        v37 = 0;
      }
      if ( v31 > v37 && *(_QWORD *)(v46 + 8 * v36) )
        v35 = sub_B192F0(v32, v35, v38);
      v29 = v33;
      if ( v47 == v34 )
        break;
      v33 = v34++;
    }
    v22 = v31;
    a1 = v32;
    v7 = v44;
    v27 = v35;
    v21 = v45;
    v26 = v46;
    if ( v35 )
    {
LABEL_57:
      v39 = (unsigned int)(*(_DWORD *)(v27 + 44) + 1);
      v40 = *(_DWORD *)(v27 + 44) + 1;
    }
    else
    {
      v39 = 0;
      v40 = 0;
    }
    v41 = 0;
    if ( v22 > v40 )
      v41 = *(_QWORD *)(v26 + 8 * v39);
    *(_BYTE *)(a1 + 112) = 0;
    v15 = sub_B1B5D0(a1, a2, v41);
    if ( v21 )
    {
      v42 = 0;
      v43 = (unsigned int)(*(_DWORD *)(v7 + 44) + 1);
      if ( (unsigned int)v43 < *(_DWORD *)(a1 + 32) )
        v42 = *(_QWORD **)(*(_QWORD *)(a1 + 24) + 8 * v43);
      *(_BYTE *)(a1 + 112) = 0;
      sub_B1AE50(v42, v15);
    }
    if ( v49 != v51 )
      goto LABEL_62;
  }
  else
  {
LABEL_69:
    if ( v49 != v51 )
LABEL_62:
      _libc_free(v49, v15);
  }
}
