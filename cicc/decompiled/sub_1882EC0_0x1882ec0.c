// Function: sub_1882EC0
// Address: 0x1882ec0
//
void __fastcall sub_1882EC0(__int64 *a1, unsigned __int64 a2)
{
  __int64 v3; // r15
  __int64 v4; // rbx
  char *v5; // rdx
  unsigned __int64 v6; // rax
  bool v7; // cf
  unsigned __int64 v8; // rax
  char *v9; // rbx
  __int64 v10; // rbx
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // r12
  __int64 v17; // r13
  __int64 v18; // rdi
  __int64 v19; // rdi
  __int64 v20; // rdi
  __int64 v21; // rdi
  __int64 v22; // rdi
  __int64 v23; // r12
  __int64 v24; // r13
  __int64 v25; // rdi
  __int64 v26; // rsi
  __int64 v27; // r12
  __int64 v28; // rax
  __int64 v29; // [rsp+8h] [rbp-58h]
  unsigned __int64 v30; // [rsp+10h] [rbp-50h]
  __int64 v31; // [rsp+18h] [rbp-48h]
  unsigned __int64 v32; // [rsp+20h] [rbp-40h]
  __int64 v33; // [rsp+28h] [rbp-38h]

  v32 = a2;
  if ( !a2 )
    return;
  v3 = *a1;
  v33 = a1[1];
  v4 = v33 - *a1;
  v30 = 0x86BCA1AF286BCA1BLL * (v4 >> 3);
  if ( 0x86BCA1AF286BCA1BLL * ((a1[2] - v33) >> 3) >= a2 )
  {
    v5 = (char *)a1[1];
    do
    {
      if ( v5 )
        memset(v5, 0, 0x98u);
      v5 += 152;
      --a2;
    }
    while ( a2 );
    a1[1] = v33 + 152 * v32;
    return;
  }
  if ( 0xD79435E50D7943LL - v30 < a2 )
    sub_4262D8((__int64)"vector::_M_default_append");
  v6 = a2;
  if ( v30 >= a2 )
    v6 = 0x86BCA1AF286BCA1BLL * ((v33 - *a1) >> 3);
  v7 = __CFADD__(v30, v6);
  v8 = v30 + v6;
  if ( v7 )
  {
    v27 = 0x7FFFFFFFFFFFFFC8LL;
  }
  else
  {
    if ( !v8 )
    {
      v29 = 0;
      v31 = 0;
      goto LABEL_15;
    }
    if ( v8 > 0xD79435E50D7943LL )
      v8 = 0xD79435E50D7943LL;
    v27 = 152 * v8;
  }
  v28 = sub_22077B0(v27);
  v3 = *a1;
  v31 = v28;
  v33 = a1[1];
  v29 = v28 + v27;
LABEL_15:
  v9 = (char *)(v31 + v4);
  do
  {
    if ( v9 )
      memset(v9, 0, 0x98u);
    v9 += 152;
    --a2;
  }
  while ( a2 );
  if ( v33 != v3 )
  {
    v10 = v31;
    do
    {
      if ( v10 )
      {
        *(_DWORD *)v10 = *(_DWORD *)v3;
        *(_BYTE *)(v10 + 4) = *(_BYTE *)(v3 + 4);
        *(_BYTE *)(v10 + 5) = *(_BYTE *)(v3 + 5);
        *(_BYTE *)(v10 + 6) = *(_BYTE *)(v3 + 6);
        *(_QWORD *)(v10 + 8) = *(_QWORD *)(v3 + 8);
        *(_QWORD *)(v10 + 16) = *(_QWORD *)(v3 + 16);
        *(_QWORD *)(v10 + 24) = *(_QWORD *)(v3 + 24);
        v11 = *(_QWORD *)(v3 + 32);
        *(_QWORD *)(v3 + 24) = 0;
        *(_QWORD *)(v3 + 16) = 0;
        *(_QWORD *)(v3 + 8) = 0;
        *(_QWORD *)(v10 + 32) = v11;
        *(_QWORD *)(v10 + 40) = *(_QWORD *)(v3 + 40);
        *(_QWORD *)(v10 + 48) = *(_QWORD *)(v3 + 48);
        v12 = *(_QWORD *)(v3 + 56);
        *(_QWORD *)(v3 + 48) = 0;
        *(_QWORD *)(v3 + 40) = 0;
        *(_QWORD *)(v3 + 32) = 0;
        *(_QWORD *)(v10 + 56) = v12;
        *(_QWORD *)(v10 + 64) = *(_QWORD *)(v3 + 64);
        *(_QWORD *)(v10 + 72) = *(_QWORD *)(v3 + 72);
        *(_QWORD *)(v3 + 72) = 0;
        v13 = *(_QWORD *)(v3 + 80);
        *(_QWORD *)(v3 + 64) = 0;
        *(_QWORD *)(v3 + 56) = 0;
        *(_QWORD *)(v10 + 80) = v13;
        *(_QWORD *)(v10 + 88) = *(_QWORD *)(v3 + 88);
        *(_QWORD *)(v10 + 96) = *(_QWORD *)(v3 + 96);
        v14 = *(_QWORD *)(v3 + 104);
        *(_QWORD *)(v3 + 96) = 0;
        *(_QWORD *)(v3 + 88) = 0;
        *(_QWORD *)(v3 + 80) = 0;
        *(_QWORD *)(v10 + 104) = v14;
        *(_QWORD *)(v10 + 112) = *(_QWORD *)(v3 + 112);
        *(_QWORD *)(v10 + 120) = *(_QWORD *)(v3 + 120);
        v15 = *(_QWORD *)(v3 + 128);
        *(_QWORD *)(v3 + 120) = 0;
        *(_QWORD *)(v3 + 112) = 0;
        *(_QWORD *)(v3 + 104) = 0;
        *(_QWORD *)(v10 + 128) = v15;
        *(_QWORD *)(v10 + 136) = *(_QWORD *)(v3 + 136);
        *(_QWORD *)(v10 + 144) = *(_QWORD *)(v3 + 144);
        *(_QWORD *)(v3 + 144) = 0;
        *(_QWORD *)(v3 + 136) = 0;
        *(_QWORD *)(v3 + 128) = 0;
      }
      else
      {
        v23 = *(_QWORD *)(v3 + 136);
        v24 = *(_QWORD *)(v3 + 128);
        if ( v23 == v24 )
        {
          v26 = *(_QWORD *)(v3 + 144) - v24;
        }
        else
        {
          do
          {
            v25 = *(_QWORD *)(v24 + 16);
            if ( v25 )
              j_j___libc_free_0(v25, *(_QWORD *)(v24 + 32) - v25);
            v24 += 40;
          }
          while ( v23 != v24 );
          v24 = *(_QWORD *)(v3 + 128);
          v26 = *(_QWORD *)(v3 + 144) - v24;
        }
        if ( v24 )
          j_j___libc_free_0(v24, v26);
      }
      v16 = *(_QWORD *)(v3 + 112);
      v17 = *(_QWORD *)(v3 + 104);
      if ( v16 != v17 )
      {
        do
        {
          v18 = *(_QWORD *)(v17 + 16);
          if ( v18 )
            j_j___libc_free_0(v18, *(_QWORD *)(v17 + 32) - v18);
          v17 += 40;
        }
        while ( v16 != v17 );
        v17 = *(_QWORD *)(v3 + 104);
      }
      if ( v17 )
        j_j___libc_free_0(v17, *(_QWORD *)(v3 + 120) - v17);
      v19 = *(_QWORD *)(v3 + 80);
      if ( v19 )
        j_j___libc_free_0(v19, *(_QWORD *)(v3 + 96) - v19);
      v20 = *(_QWORD *)(v3 + 56);
      if ( v20 )
        j_j___libc_free_0(v20, *(_QWORD *)(v3 + 72) - v20);
      v21 = *(_QWORD *)(v3 + 32);
      if ( v21 )
        j_j___libc_free_0(v21, *(_QWORD *)(v3 + 48) - v21);
      v22 = *(_QWORD *)(v3 + 8);
      if ( v22 )
        j_j___libc_free_0(v22, *(_QWORD *)(v3 + 24) - v22);
      v3 += 152;
      v10 += 152;
    }
    while ( v3 != v33 );
    v3 = *a1;
  }
  if ( v3 )
    j_j___libc_free_0(v3, a1[2] - v3);
  *a1 = v31;
  a1[1] = v31 + 152 * (v32 + v30);
  a1[2] = v29;
}
