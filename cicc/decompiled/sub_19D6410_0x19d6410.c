// Function: sub_19D6410
// Address: 0x19d6410
//
__int64 *__fastcall sub_19D6410(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rbx
  __int64 v5; // r12
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v9; // r14
  bool v10; // cf
  unsigned __int64 v11; // rax
  __int64 v12; // rsi
  __int64 v13; // r15
  __int64 v14; // rax
  unsigned int v15; // esi
  unsigned int v16; // esi
  __int64 v17; // r15
  __int64 v18; // rax
  unsigned int v19; // esi
  unsigned int v20; // esi
  unsigned int v21; // eax
  int v22; // eax
  unsigned int v23; // eax
  __int64 i; // r13
  __int64 v25; // rdi
  __int64 v26; // rdi
  __int64 v28; // r15
  __int64 v29; // rax
  __int64 v30; // [rsp+8h] [rbp-58h]
  __int64 v31; // [rsp+8h] [rbp-58h]
  __int64 v32; // [rsp+8h] [rbp-58h]
  __int64 v33; // [rsp+10h] [rbp-50h]
  __int64 v35; // [rsp+20h] [rbp-40h]
  __int64 v36; // [rsp+20h] [rbp-40h]
  __int64 v37; // [rsp+20h] [rbp-40h]
  __int64 v38; // [rsp+20h] [rbp-40h]
  __int64 v39; // [rsp+28h] [rbp-38h]

  v4 = a1[1];
  v5 = *a1;
  v6 = 0x4EC4EC4EC4EC4EC5LL * ((v4 - *a1) >> 3);
  if ( v6 == 0x13B13B13B13B13BLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v7 = 1;
  v9 = a2;
  if ( v6 )
    v7 = 0x4EC4EC4EC4EC4EC5LL * ((v4 - v5) >> 3);
  v10 = __CFADD__(v7, v6);
  v11 = v7 + v6;
  v12 = a2 - v5;
  if ( v10 )
  {
    v28 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v11 )
    {
      v33 = 0;
      v13 = 104;
      v39 = 0;
      goto LABEL_7;
    }
    if ( v11 > 0x13B13B13B13B13BLL )
      v11 = 0x13B13B13B13B13BLL;
    v28 = 104 * v11;
  }
  v30 = a3;
  v29 = sub_22077B0(v28);
  a3 = v30;
  v39 = v29;
  v33 = v29 + v28;
  v13 = v29 + 104;
LABEL_7:
  v14 = v39 + v12;
  if ( v39 + v12 )
  {
    *(_QWORD *)v14 = *(_QWORD *)a3;
    *(_QWORD *)(v14 + 8) = *(_QWORD *)(a3 + 8);
    *(_QWORD *)(v14 + 16) = *(_QWORD *)(a3 + 16);
    *(_BYTE *)(v14 + 24) = *(_BYTE *)(a3 + 24);
    *(_QWORD *)(v14 + 32) = *(_QWORD *)(a3 + 32);
    *(_QWORD *)(v14 + 40) = *(_QWORD *)(a3 + 40);
    v15 = *(_DWORD *)(a3 + 56);
    *(_DWORD *)(v14 + 56) = v15;
    if ( v15 > 0x40 )
    {
      v31 = a3;
      v37 = v14;
      sub_16A4FD0(v14 + 48, (const void **)(a3 + 48));
      a3 = v31;
      v14 = v37;
    }
    else
    {
      *(_QWORD *)(v14 + 48) = *(_QWORD *)(a3 + 48);
    }
    *(_QWORD *)(v14 + 64) = *(_QWORD *)(a3 + 64);
    *(_QWORD *)(v14 + 72) = *(_QWORD *)(a3 + 72);
    v16 = *(_DWORD *)(a3 + 88);
    *(_DWORD *)(v14 + 88) = v16;
    if ( v16 > 0x40 )
    {
      v32 = a3;
      v38 = v14;
      sub_16A4FD0(v14 + 80, (const void **)(a3 + 80));
      a3 = v32;
      v14 = v38;
    }
    else
    {
      *(_QWORD *)(v14 + 80) = *(_QWORD *)(a3 + 80);
    }
    *(_DWORD *)(v14 + 96) = *(_DWORD *)(a3 + 96);
  }
  if ( a2 != v5 )
  {
    v17 = v39;
    v18 = v5;
    while ( 1 )
    {
      if ( v17 )
      {
        *(_QWORD *)v17 = *(_QWORD *)v18;
        *(_QWORD *)(v17 + 8) = *(_QWORD *)(v18 + 8);
        *(_QWORD *)(v17 + 16) = *(_QWORD *)(v18 + 16);
        *(_BYTE *)(v17 + 24) = *(_BYTE *)(v18 + 24);
        *(_QWORD *)(v17 + 32) = *(_QWORD *)(v18 + 32);
        *(_QWORD *)(v17 + 40) = *(_QWORD *)(v18 + 40);
        v20 = *(_DWORD *)(v18 + 56);
        *(_DWORD *)(v17 + 56) = v20;
        if ( v20 <= 0x40 )
        {
          *(_QWORD *)(v17 + 48) = *(_QWORD *)(v18 + 48);
        }
        else
        {
          v35 = v18;
          sub_16A4FD0(v17 + 48, (const void **)(v18 + 48));
          v18 = v35;
        }
        *(_QWORD *)(v17 + 64) = *(_QWORD *)(v18 + 64);
        *(_QWORD *)(v17 + 72) = *(_QWORD *)(v18 + 72);
        v19 = *(_DWORD *)(v18 + 88);
        *(_DWORD *)(v17 + 88) = v19;
        if ( v19 > 0x40 )
        {
          v36 = v18;
          sub_16A4FD0(v17 + 80, (const void **)(v18 + 80));
          v18 = v36;
        }
        else
        {
          *(_QWORD *)(v17 + 80) = *(_QWORD *)(v18 + 80);
        }
        *(_DWORD *)(v17 + 96) = *(_DWORD *)(v18 + 96);
      }
      v18 += 104;
      if ( a2 == v18 )
        break;
      v17 += 104;
    }
    v13 = v17 + 208;
  }
  if ( a2 != v4 )
  {
    do
    {
      *(_QWORD *)v13 = *(_QWORD *)v9;
      *(_QWORD *)(v13 + 8) = *(_QWORD *)(v9 + 8);
      *(_QWORD *)(v13 + 16) = *(_QWORD *)(v9 + 16);
      *(_BYTE *)(v13 + 24) = *(_BYTE *)(v9 + 24);
      *(_QWORD *)(v13 + 32) = *(_QWORD *)(v9 + 32);
      *(_QWORD *)(v13 + 40) = *(_QWORD *)(v9 + 40);
      v23 = *(_DWORD *)(v9 + 56);
      *(_DWORD *)(v13 + 56) = v23;
      if ( v23 <= 0x40 )
        *(_QWORD *)(v13 + 48) = *(_QWORD *)(v9 + 48);
      else
        sub_16A4FD0(v13 + 48, (const void **)(v9 + 48));
      *(_QWORD *)(v13 + 64) = *(_QWORD *)(v9 + 64);
      *(_QWORD *)(v13 + 72) = *(_QWORD *)(v9 + 72);
      v21 = *(_DWORD *)(v9 + 88);
      *(_DWORD *)(v13 + 88) = v21;
      if ( v21 > 0x40 )
        sub_16A4FD0(v13 + 80, (const void **)(v9 + 80));
      else
        *(_QWORD *)(v13 + 80) = *(_QWORD *)(v9 + 80);
      v22 = *(_DWORD *)(v9 + 96);
      v9 += 104;
      v13 += 104;
      *(_DWORD *)(v13 - 8) = v22;
    }
    while ( v4 != v9 );
  }
  for ( i = v5; v4 != i; i += 104 )
  {
    if ( *(_DWORD *)(i + 88) > 0x40u )
    {
      v25 = *(_QWORD *)(i + 80);
      if ( v25 )
        j_j___libc_free_0_0(v25);
    }
    if ( *(_DWORD *)(i + 56) > 0x40u )
    {
      v26 = *(_QWORD *)(i + 48);
      if ( v26 )
        j_j___libc_free_0_0(v26);
    }
  }
  if ( v5 )
    j_j___libc_free_0(v5, a1[2] - v5);
  *a1 = v39;
  a1[1] = v13;
  a1[2] = v33;
  return a1;
}
