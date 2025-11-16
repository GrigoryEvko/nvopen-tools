// Function: sub_25729F0
// Address: 0x25729f0
//
void __fastcall sub_25729F0(__int64 a1, __int64 a2)
{
  __int64 v3; // r12
  unsigned __int8 v4; // al
  __int64 v5; // rax
  __int64 v6; // rdi
  __int64 v7; // r13
  __int64 v8; // r15
  __int64 *v9; // rbx
  __int64 *v10; // r12
  __int64 v11; // rdx
  __int64 v12; // rcx
  int v13; // r10d
  unsigned int j; // eax
  __int64 v15; // rdi
  __int64 v16; // rsi
  __int64 v17; // rax
  unsigned __int64 v18; // rdx
  __int64 v19; // rbx
  unsigned __int64 v20; // rax
  __int64 v21; // r13
  int v22; // r12d
  unsigned int v23; // r15d
  __int64 v24; // rcx
  __int64 v25; // r8
  int v26; // ebx
  unsigned int i; // edx
  __int64 v28; // r9
  unsigned int v29; // edx
  unsigned int v30; // eax
  __int64 v31; // r15
  __int64 *v32; // [rsp+10h] [rbp-A0h] BYREF
  __int64 v33; // [rsp+18h] [rbp-98h]
  __int64 v34[2]; // [rsp+20h] [rbp-90h] BYREF
  __int64 v35; // [rsp+30h] [rbp-80h]
  unsigned __int64 v36; // [rsp+38h] [rbp-78h]
  __int64 v37; // [rsp+40h] [rbp-70h]
  __int64 v38; // [rsp+48h] [rbp-68h]
  unsigned __int64 v39; // [rsp+50h] [rbp-60h]
  __int64 v40; // [rsp+58h] [rbp-58h]
  __int64 v41; // [rsp+60h] [rbp-50h]
  unsigned __int64 v42; // [rsp+68h] [rbp-48h]
  __int64 v43; // [rsp+70h] [rbp-40h]
  __int64 v44; // [rsp+78h] [rbp-38h]

  v3 = *(_QWORD *)(a1 + 72) & 0xFFFFFFFFFFFFFFFCLL;
  if ( (*(_QWORD *)(a1 + 72) & 3LL) == 3 )
    v3 = *(_QWORD *)(v3 + 24);
  v4 = *(_BYTE *)v3;
  if ( *(_BYTE *)v3 )
  {
    if ( v4 == 22 )
    {
      v3 = *(_QWORD *)(v3 + 24);
    }
    else if ( v4 <= 0x1Cu )
    {
      v3 = 0;
    }
    else
    {
      v3 = sub_B43CB0(v3);
    }
  }
  if ( sub_B2FC80(v3) )
    goto LABEL_8;
  v5 = *(_QWORD *)(*(_QWORD *)(a2 + 208) + 240LL);
  v6 = *(_QWORD *)v5;
  if ( !*(_QWORD *)v5 )
    goto LABEL_28;
  if ( *(_BYTE *)(v5 + 16) )
  {
    v24 = *(unsigned int *)(v6 + 88);
    v25 = *(_QWORD *)(v6 + 72);
    if ( !(_DWORD)v24 )
      goto LABEL_64;
    v26 = 1;
    for ( i = (v24 - 1)
            & (((0xBF58476D1CE4E5B9LL
               * (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4)
                | ((unsigned __int64)(((unsigned int)&unk_4F881D0 >> 9) ^ ((unsigned int)&unk_4F881D0 >> 4)) << 32))) >> 31)
             ^ (484763065 * (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4)))); ; i = (v24 - 1) & v29 )
    {
      v28 = v25 + 24LL * i;
      if ( *(_UNKNOWN **)v28 == &unk_4F881D0 && v3 == *(_QWORD *)(v28 + 8) )
        break;
      if ( *(_QWORD *)v28 == -4096 && *(_QWORD *)(v28 + 8) == -4096 )
        goto LABEL_64;
      v29 = v26 + i;
      ++v26;
    }
    if ( v28 == v25 + 24 * v24 )
    {
LABEL_64:
      v7 = 0;
    }
    else
    {
      v7 = *(_QWORD *)(*(_QWORD *)(v28 + 16) + 24LL);
      if ( v7 )
        v7 += 8;
    }
  }
  else
  {
    v7 = sub_BC1CD0(v6, &unk_4F881D0, v3) + 8;
    v5 = *(_QWORD *)(*(_QWORD *)(a2 + 208) + 240LL);
    v6 = *(_QWORD *)v5;
    if ( !*(_QWORD *)v5 )
      goto LABEL_28;
  }
  if ( *(_BYTE *)(v5 + 16) )
  {
    v11 = *(unsigned int *)(v6 + 88);
    v12 = *(_QWORD *)(v6 + 72);
    if ( !(_DWORD)v11 )
      goto LABEL_28;
    v13 = 1;
    for ( j = (v11 - 1)
            & (((0xBF58476D1CE4E5B9LL
               * (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4)
                | ((unsigned __int64)(((unsigned int)&unk_4F875F0 >> 9) ^ ((unsigned int)&unk_4F875F0 >> 4)) << 32))) >> 31)
             ^ (484763065 * (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4)))); ; j = (v11 - 1) & v30 )
    {
      v15 = v12 + 24LL * j;
      if ( *(_UNKNOWN **)v15 == &unk_4F875F0 && v3 == *(_QWORD *)(v15 + 8) )
        break;
      if ( *(_QWORD *)v15 == -4096 && *(_QWORD *)(v15 + 8) == -4096 )
        goto LABEL_28;
      v30 = v13 + j;
      ++v13;
    }
    if ( v15 == v12 + 24 * v11 )
      goto LABEL_28;
    v31 = *(_QWORD *)(*(_QWORD *)(v15 + 16) + 24LL);
    if ( !v31 )
      goto LABEL_28;
    v8 = v31 + 8;
  }
  else
  {
    v8 = sub_BC1CD0(v6, &unk_4F875F0, v3) + 8;
  }
  if ( !v7 )
  {
LABEL_28:
    v16 = *(_QWORD *)(v3 + 80);
    v35 = 0;
    v32 = 0;
    v33 = 0;
    if ( v16 )
      v16 -= 24;
    v34[0] = 0;
    v34[1] = 0;
    v36 = 0;
    v37 = 0;
    v38 = 0;
    v39 = 0;
    v40 = 0;
    v41 = 0;
    v42 = 0;
    v43 = 0;
    v44 = 0;
    sub_2572230((__int64)&v32, v16);
    sub_25725D0((__int64)&v32);
    v17 = v40;
    v18 = v39;
    if ( v39 != v40 )
    {
      while ( v17 - v18 <= 8 )
      {
        v19 = *(_QWORD *)v18;
        v20 = *(_QWORD *)(*(_QWORD *)v18 + 48LL) & 0xFFFFFFFFFFFFFFF8LL;
        if ( v20 != *(_QWORD *)v18 + 48LL )
        {
          if ( !v20 )
            BUG();
          v21 = v20 - 24;
          if ( (unsigned int)*(unsigned __int8 *)(v20 - 24) - 30 <= 0xA )
          {
            v22 = sub_B46E30(v21);
            if ( v22 )
            {
              v23 = 0;
              while ( v19 != sub_B46EC0(v21, v23) )
              {
                if ( v22 == ++v23 )
                  goto LABEL_46;
              }
              break;
            }
          }
        }
LABEL_46:
        sub_25725D0((__int64)&v32);
        v17 = v40;
        v18 = v39;
        if ( v40 == v39 )
          goto LABEL_47;
      }
      if ( v42 )
        j_j___libc_free_0(v42);
      if ( v39 )
        j_j___libc_free_0(v39);
      if ( v36 )
        j_j___libc_free_0(v36);
      sub_C7D6A0(v34[0], 16LL * (unsigned int)v35, 8);
      goto LABEL_8;
    }
LABEL_47:
    if ( v42 )
      j_j___libc_free_0(v42);
    if ( v39 )
      j_j___libc_free_0(v39);
    if ( v36 )
      j_j___libc_free_0(v36);
    sub_C7D6A0(v34[0], 16LL * (unsigned int)v35, 8);
    return;
  }
  if ( (unsigned __int8)sub_31052D0(v3, v8) )
  {
LABEL_8:
    *(_BYTE *)(a1 + 97) = *(_BYTE *)(a1 + 96);
    return;
  }
  sub_D47CF0(&v32, v8);
  v9 = v32;
  v10 = &v32[(unsigned int)v33];
  if ( v32 != v10 )
  {
    while ( (unsigned int)sub_DBB070(v7, *v9, 0) )
    {
      if ( v10 == ++v9 )
      {
        v10 = v32;
        goto LABEL_66;
      }
    }
    if ( v32 != v34 )
      _libc_free((unsigned __int64)v32);
    goto LABEL_8;
  }
LABEL_66:
  if ( v10 != v34 )
    _libc_free((unsigned __int64)v10);
}
