// Function: sub_D33160
// Address: 0xd33160
//
__int64 __fastcall sub_D33160(__int64 *a1)
{
  __int64 v2; // rsi
  __int64 v3; // rdi
  __int64 *v4; // rdi
  __int64 v5; // r15
  __int64 v6; // rsi
  __int64 v7; // rdi
  __int64 v8; // rdi
  __int64 v9; // rsi
  _QWORD *v10; // r12
  _QWORD *v11; // r14
  __int64 v12; // rdi
  __int64 v13; // r14
  __int64 v14; // rdi
  __int64 v15; // rdi
  __int64 v16; // r15
  __int64 v17; // r12
  __int64 v18; // rdi
  _QWORD *v19; // r15
  _QWORD *v20; // r12
  __int64 v21; // rax
  __int64 v22; // r14
  __int64 v23; // r12
  __int64 v24; // rdi
  __int64 v25; // rax
  _QWORD *v27; // r12
  _QWORD *v28; // rbx
  __int64 v29; // rax
  __int64 v30; // rsi
  __int64 v31; // rax
  _QWORD *v32; // r12
  _QWORD *v33; // rbx
  __int64 v34; // rsi
  _QWORD v35[2]; // [rsp+8h] [rbp-88h] BYREF
  __int64 v36; // [rsp+18h] [rbp-78h]
  __int64 v37; // [rsp+20h] [rbp-70h]
  void *v38; // [rsp+30h] [rbp-60h]
  _QWORD v39[2]; // [rsp+38h] [rbp-58h] BYREF
  __int64 v40; // [rsp+48h] [rbp-48h]
  __int64 v41; // [rsp+50h] [rbp-40h]

  v2 = 16LL * *((unsigned int *)a1 + 36);
  sub_C7D6A0(a1[16], v2, 8);
  v3 = a1[14];
  if ( v3 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v3 + 16LL))(v3);
  v4 = (__int64 *)a1[6];
  if ( v4 != a1 + 8 )
    _libc_free(v4, v2);
  v5 = a1[2];
  if ( v5 )
  {
    if ( *(_BYTE *)(v5 + 440) )
    {
      v30 = *(unsigned int *)(v5 + 416);
      *(_BYTE *)(v5 + 440) = 0;
      sub_C7D6A0(*(_QWORD *)(v5 + 400), 16 * v30, 8);
    }
    v6 = 32LL * *(unsigned int *)(v5 + 384);
    sub_C7D6A0(*(_QWORD *)(v5 + 368), v6, 8);
    v7 = *(_QWORD *)(v5 + 240);
    if ( v7 != v5 + 256 )
      _libc_free(v7, v6);
    v8 = *(_QWORD *)(v5 + 56);
    if ( v8 != v5 + 72 )
      _libc_free(v8, v6);
    v9 = *(unsigned int *)(v5 + 48);
    if ( (_DWORD)v9 )
    {
      v10 = *(_QWORD **)(v5 + 32);
      v11 = &v10[4 * v9];
      do
      {
        if ( *v10 != -16 && *v10 != -4 )
        {
          v12 = v10[1];
          if ( v12 )
            j_j___libc_free_0(v12, v10[3] - v12);
        }
        v10 += 4;
      }
      while ( v11 != v10 );
      v9 = *(unsigned int *)(v5 + 48);
    }
    sub_C7D6A0(*(_QWORD *)(v5 + 32), 32 * v9, 8);
    v2 = 448;
    j_j___libc_free_0(v5, 448);
  }
  v13 = a1[1];
  if ( v13 )
  {
    v14 = *(_QWORD *)(v13 + 384);
    if ( v14 != v13 + 400 )
      _libc_free(v14, v2);
    v15 = *(_QWORD *)(v13 + 296);
    if ( v15 != v13 + 312 )
      _libc_free(v15, v2);
    v16 = *(_QWORD *)(v13 + 168);
    v17 = v16 + 48LL * *(unsigned int *)(v13 + 176);
    if ( v16 != v17 )
    {
      do
      {
        v17 -= 48;
        v18 = *(_QWORD *)(v17 + 16);
        if ( v18 != v17 + 32 )
          _libc_free(v18, v2);
      }
      while ( v16 != v17 );
      v17 = *(_QWORD *)(v13 + 168);
    }
    if ( v17 != v13 + 184 )
      _libc_free(v17, v2);
    v19 = *(_QWORD **)(v13 + 8);
    v20 = &v19[9 * *(unsigned int *)(v13 + 16)];
    if ( v19 != v20 )
    {
      do
      {
        v21 = *(v20 - 7);
        v20 -= 9;
        if ( v21 != 0 && v21 != -4096 && v21 != -8192 )
          sub_BD60C0(v20);
      }
      while ( v19 != v20 );
      v20 = *(_QWORD **)(v13 + 8);
    }
    if ( v20 != (_QWORD *)(v13 + 24) )
      _libc_free(v20, v2);
    v2 = 448;
    j_j___libc_free_0(v13, 448);
  }
  v22 = *a1;
  if ( *a1 )
  {
    v23 = *(_QWORD *)(v22 + 128);
    if ( v23 )
    {
      v24 = *(_QWORD *)(v23 + 40);
      if ( v24 != v23 + 56 )
        _libc_free(v24, v2);
      j_j___libc_free_0(v23, 184);
    }
    if ( *(_BYTE *)(v22 + 96) )
    {
      v31 = *(unsigned int *)(v22 + 88);
      *(_BYTE *)(v22 + 96) = 0;
      if ( (_DWORD)v31 )
      {
        v32 = *(_QWORD **)(v22 + 72);
        v33 = &v32[2 * v31];
        do
        {
          if ( *v32 != -4096 && *v32 != -8192 )
          {
            v34 = v32[1];
            if ( v34 )
              sub_B91220((__int64)(v32 + 1), v34);
          }
          v32 += 2;
        }
        while ( v33 != v32 );
        LODWORD(v31) = *(_DWORD *)(v22 + 88);
      }
      sub_C7D6A0(*(_QWORD *)(v22 + 72), 16LL * (unsigned int)v31, 8);
      v25 = *(unsigned int *)(v22 + 56);
      if ( !(_DWORD)v25 )
        goto LABEL_49;
    }
    else
    {
      v25 = *(unsigned int *)(v22 + 56);
      if ( !(_DWORD)v25 )
      {
LABEL_49:
        sub_C7D6A0(*(_QWORD *)(v22 + 40), 48 * v25, 8);
        sub_C7D6A0(*(_QWORD *)(v22 + 8), 24LL * *(unsigned int *)(v22 + 24), 8);
        j_j___libc_free_0(v22, 168);
        return j_j___libc_free_0(a1, 152);
      }
    }
    v27 = *(_QWORD **)(v22 + 40);
    v35[0] = 2;
    v35[1] = 0;
    v36 = -4096;
    v28 = &v27[6 * v25];
    v37 = 0;
    v39[0] = 2;
    v39[1] = 0;
    v40 = -8192;
    v38 = &unk_49DDFA0;
    v41 = 0;
    do
    {
      v29 = v27[3];
      *v27 = &unk_49DB368;
      if ( v29 != 0 && v29 != -4096 && v29 != -8192 )
        sub_BD60C0(v27 + 1);
      v27 += 6;
    }
    while ( v28 != v27 );
    v38 = &unk_49DB368;
    if ( v40 != -4096 && v40 != 0 && v40 != -8192 )
      sub_BD60C0(v39);
    if ( v36 != -4096 && v36 != 0 && v36 != -8192 )
      sub_BD60C0(v35);
    v25 = *(unsigned int *)(v22 + 56);
    goto LABEL_49;
  }
  return j_j___libc_free_0(a1, 152);
}
