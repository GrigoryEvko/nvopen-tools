// Function: sub_163C720
// Address: 0x163c720
//
void __fastcall sub_163C720(__int64 a1, char a2)
{
  __int64 v3; // rax
  __int64 v4; // rax
  _QWORD *v5; // rbx
  _QWORD *v6; // r12
  _QWORD *v7; // rax
  _QWORD *v8; // rax
  const void *v9; // rsi
  void *v10; // rax
  __int64 v11; // rdx
  const void *v12; // rsi
  unsigned int v13; // esi
  unsigned int v14; // eax
  _QWORD *v15; // rdi
  __int64 v16; // rcx
  int v17; // eax
  _QWORD *v18; // r10
  __int64 v19; // rcx
  int v20; // edx
  int v21; // r11d
  _QWORD *v22; // r9
  int v23; // r11d
  int v24; // edi
  _QWORD *v25; // rsi
  int v26; // eax
  __int64 v27; // r15
  __int64 v28; // [rsp+0h] [rbp-50h] BYREF
  _QWORD *v29; // [rsp+8h] [rbp-48h]
  __int64 v30; // [rsp+10h] [rbp-40h]
  __int64 v31; // [rsp+18h] [rbp-38h]

  if ( !*(_BYTE *)(a1 + 96) )
  {
    v28 = 0;
    v29 = 0;
    v30 = 0;
    v31 = 0;
    j___libc_free_0(0);
    v4 = *(unsigned int *)(a1 + 88);
    LODWORD(v31) = v4;
    if ( (_DWORD)v4 )
    {
      v8 = (_QWORD *)sub_22077B0(8 * v4);
      v9 = *(const void **)(a1 + 72);
      v29 = v8;
      v30 = *(_QWORD *)(a1 + 80);
      memcpy(v8, v9, 8LL * (unsigned int)v31);
    }
    else
    {
      v29 = 0;
      v30 = 0;
    }
    v5 = *(_QWORD **)(a1 + 8);
    v6 = &v5[*(unsigned int *)(a1 + 24)];
    if ( !*(_DWORD *)(a1 + 16) || v6 == v5 )
      goto LABEL_8;
    while ( *v5 == -16 || *v5 == -8 )
    {
      if ( ++v5 == v6 )
        goto LABEL_8;
    }
    if ( v5 == v6 )
    {
LABEL_8:
      j___libc_free_0(*(_QWORD *)(a1 + 40));
      v7 = v29;
      ++*(_QWORD *)(a1 + 32);
      ++v28;
      *(_QWORD *)(a1 + 40) = v7;
      v29 = 0;
      *(_QWORD *)(a1 + 48) = v30;
      v30 = 0;
      *(_DWORD *)(a1 + 56) = v31;
      j___libc_free_0(0);
      return;
    }
    v13 = v31;
    if ( !(_DWORD)v31 )
      goto LABEL_24;
    while ( 1 )
    {
      v14 = (v13 - 1) & (((unsigned int)*v5 >> 9) ^ ((unsigned int)*v5 >> 4));
      v15 = &v29[v14];
      v16 = *v15;
      if ( *v5 != *v15 )
      {
        v23 = 1;
        v18 = 0;
        while ( v16 != -8 )
        {
          if ( v18 || v16 != -16 )
            v15 = v18;
          v14 = (v13 - 1) & (v23 + v14);
          v16 = v29[v14];
          if ( *v5 == v16 )
            goto LABEL_19;
          ++v23;
          v18 = v15;
          v15 = &v29[v14];
        }
        if ( !v18 )
          v18 = v15;
        ++v28;
        v20 = v30 + 1;
        if ( 4 * ((int)v30 + 1) >= 3 * v13 )
          goto LABEL_25;
        if ( v13 - HIDWORD(v30) - v20 <= v13 >> 3 )
        {
          sub_13B3B90((__int64)&v28, v13);
          if ( !(_DWORD)v31 )
          {
LABEL_62:
            LODWORD(v30) = v30 + 1;
            BUG();
          }
          v24 = 1;
          v25 = 0;
          v26 = (v31 - 1) & (((unsigned int)*v5 >> 9) ^ ((unsigned int)*v5 >> 4));
          v18 = &v29[v26];
          v27 = *v18;
          v20 = v30 + 1;
          if ( *v18 != *v5 )
          {
            while ( v27 != -8 )
            {
              if ( !v25 && v27 == -16 )
                v25 = v18;
              v26 = (v31 - 1) & (v24 + v26);
              v18 = &v29[v26];
              v27 = *v18;
              if ( *v5 == *v18 )
                goto LABEL_38;
              ++v24;
            }
            if ( v25 )
              v18 = v25;
          }
        }
LABEL_38:
        LODWORD(v30) = v20;
        if ( *v18 != -8 )
          --HIDWORD(v30);
        *v18 = *v5;
      }
      do
      {
LABEL_19:
        if ( ++v5 == v6 )
          goto LABEL_8;
      }
      while ( *v5 == -16 || *v5 == -8 );
      if ( v6 == v5 )
        goto LABEL_8;
      v13 = v31;
      if ( !(_DWORD)v31 )
      {
LABEL_24:
        ++v28;
LABEL_25:
        sub_13B3B90((__int64)&v28, 2 * v13);
        if ( !(_DWORD)v31 )
          goto LABEL_62;
        v17 = (v31 - 1) & (((unsigned int)*v5 >> 9) ^ ((unsigned int)*v5 >> 4));
        v18 = &v29[v17];
        v19 = *v18;
        v20 = v30 + 1;
        if ( *v18 != *v5 )
        {
          v21 = 1;
          v22 = 0;
          while ( v19 != -8 )
          {
            if ( !v22 && v19 == -16 )
              v22 = v18;
            v17 = (v31 - 1) & (v21 + v17);
            v18 = &v29[v17];
            v19 = *v18;
            if ( *v5 == *v18 )
              goto LABEL_38;
            ++v21;
          }
          if ( v22 )
            v18 = v22;
        }
        goto LABEL_38;
      }
    }
  }
  if ( a2 )
  {
    j___libc_free_0(*(_QWORD *)(a1 + 40));
    v3 = *(unsigned int *)(a1 + 88);
    *(_DWORD *)(a1 + 56) = v3;
    if ( (_DWORD)v3 )
    {
      v10 = (void *)sub_22077B0(8 * v3);
      v11 = *(unsigned int *)(a1 + 56);
      v12 = *(const void **)(a1 + 72);
      *(_QWORD *)(a1 + 40) = v10;
      *(_QWORD *)(a1 + 48) = *(_QWORD *)(a1 + 80);
      memcpy(v10, v12, 8 * v11);
    }
    else
    {
      *(_QWORD *)(a1 + 40) = 0;
      *(_QWORD *)(a1 + 48) = 0;
    }
  }
}
