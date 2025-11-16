// Function: sub_385CEA0
// Address: 0x385cea0
//
void __fastcall sub_385CEA0(unsigned __int64 *a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // r14
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rdi
  __int64 v7; // rbx
  _QWORD *v8; // r12
  _QWORD *v9; // rbx
  unsigned __int64 v10; // rdi
  unsigned __int64 v11; // r14
  unsigned __int64 v12; // rdi
  __int64 v13; // rbx
  unsigned __int64 v14; // r12
  unsigned __int64 v15; // rdi
  _QWORD *v16; // rbx
  _QWORD *v17; // r12
  __int64 v18; // rax
  unsigned __int64 v19; // r14
  __int64 v20; // rax
  _QWORD *v21; // r12
  _QWORD *v22; // rbx
  unsigned __int64 v23; // rdi
  unsigned __int64 v24; // rdi
  __int64 v25; // rax
  _QWORD *v26; // r12
  _QWORD *v27; // r15
  __int64 v28; // rax
  __int64 v29; // rax
  _QWORD *v30; // rbx
  _QWORD *v31; // r12
  __int64 v32; // rsi
  _QWORD v33[2]; // [rsp+8h] [rbp-88h] BYREF
  __int64 v34; // [rsp+18h] [rbp-78h]
  __int64 v35; // [rsp+20h] [rbp-70h]
  void *v36; // [rsp+30h] [rbp-60h]
  _QWORD v37[2]; // [rsp+38h] [rbp-58h] BYREF
  __int64 v38; // [rsp+48h] [rbp-48h]
  __int64 v39; // [rsp+50h] [rbp-40h]

  v2 = a1[14];
  if ( v2 != a1[13] )
    _libc_free(v2);
  j___libc_free_0(a1[9]);
  v3 = a1[7];
  if ( v3 )
    (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v3 + 8LL))(v3);
  v4 = a1[2];
  if ( v4 )
  {
    v5 = *(_QWORD *)(v4 + 224);
    if ( v5 != v4 + 240 )
      _libc_free(v5);
    v6 = *(_QWORD *)(v4 + 48);
    if ( v6 != v4 + 64 )
      _libc_free(v6);
    v7 = *(unsigned int *)(v4 + 40);
    if ( (_DWORD)v7 )
    {
      v8 = *(_QWORD **)(v4 + 24);
      v9 = &v8[4 * v7];
      do
      {
        if ( *v8 != -16 && *v8 != -4 )
        {
          v10 = v8[1];
          if ( v10 )
            j_j___libc_free_0(v10);
        }
        v8 += 4;
      }
      while ( v9 != v8 );
    }
    j___libc_free_0(*(_QWORD *)(v4 + 24));
    j_j___libc_free_0(v4);
  }
  v11 = a1[1];
  if ( v11 )
  {
    v12 = *(_QWORD *)(v11 + 272);
    if ( v12 != v11 + 288 )
      _libc_free(v12);
    v13 = *(_QWORD *)(v11 + 152);
    v14 = v13 + 48LL * *(unsigned int *)(v11 + 160);
    if ( v13 != v14 )
    {
      do
      {
        v14 -= 48LL;
        v15 = *(_QWORD *)(v14 + 24);
        if ( v15 != v14 + 40 )
          _libc_free(v15);
      }
      while ( v13 != v14 );
      v14 = *(_QWORD *)(v11 + 152);
    }
    if ( v14 != v11 + 168 )
      _libc_free(v14);
    v16 = *(_QWORD **)(v11 + 8);
    v17 = &v16[8 * (unsigned __int64)*(unsigned int *)(v11 + 16)];
    if ( v16 != v17 )
    {
      do
      {
        v18 = *(v17 - 6);
        v17 -= 8;
        if ( v18 != -8 && v18 != 0 && v18 != -16 )
          sub_1649B30(v17);
      }
      while ( v16 != v17 );
      v17 = *(_QWORD **)(v11 + 8);
    }
    if ( v17 != (_QWORD *)(v11 + 24) )
      _libc_free((unsigned __int64)v17);
    j_j___libc_free_0(v11);
  }
  v19 = *a1;
  if ( *a1 )
  {
    *(_QWORD *)(v19 + 128) = &unk_49EC708;
    v20 = *(unsigned int *)(v19 + 336);
    if ( (_DWORD)v20 )
    {
      v21 = *(_QWORD **)(v19 + 320);
      v22 = &v21[7 * v20];
      do
      {
        if ( *v21 != -16 && *v21 != -8 )
        {
          v23 = v21[1];
          if ( (_QWORD *)v23 != v21 + 3 )
            _libc_free(v23);
        }
        v21 += 7;
      }
      while ( v22 != v21 );
    }
    j___libc_free_0(*(_QWORD *)(v19 + 320));
    v24 = *(_QWORD *)(v19 + 168);
    if ( v24 != v19 + 184 )
      _libc_free(v24);
    if ( *(_BYTE *)(v19 + 96) )
    {
      v29 = *(unsigned int *)(v19 + 88);
      if ( (_DWORD)v29 )
      {
        v30 = *(_QWORD **)(v19 + 72);
        v31 = &v30[2 * v29];
        do
        {
          if ( *v30 != -4 && *v30 != -8 )
          {
            v32 = v30[1];
            if ( v32 )
              sub_161E7C0((__int64)(v30 + 1), v32);
          }
          v30 += 2;
        }
        while ( v31 != v30 );
      }
      j___libc_free_0(*(_QWORD *)(v19 + 72));
      v25 = *(unsigned int *)(v19 + 56);
      if ( !(_DWORD)v25 )
        goto LABEL_49;
    }
    else
    {
      v25 = *(unsigned int *)(v19 + 56);
      if ( !(_DWORD)v25 )
      {
LABEL_49:
        j___libc_free_0(*(_QWORD *)(v19 + 40));
        j___libc_free_0(*(_QWORD *)(v19 + 8));
        j_j___libc_free_0(v19);
        goto LABEL_50;
      }
    }
    v26 = *(_QWORD **)(v19 + 40);
    v33[0] = 2;
    v33[1] = 0;
    v34 = -8;
    v27 = &v26[6 * v25];
    v35 = 0;
    v37[0] = 2;
    v37[1] = 0;
    v38 = -16;
    v36 = &unk_49EC740;
    v39 = 0;
    do
    {
      v28 = v26[3];
      *v26 = &unk_49EE2B0;
      if ( v28 != -8 && v28 != 0 && v28 != -16 )
        sub_1649B30(v26 + 1);
      v26 += 6;
    }
    while ( v27 != v26 );
    v36 = &unk_49EE2B0;
    if ( v38 != -8 && v38 != 0 && v38 != -16 )
      sub_1649B30(v37);
    if ( v34 != -8 && v34 != 0 && v34 != -16 )
      sub_1649B30(v33);
    goto LABEL_49;
  }
LABEL_50:
  j_j___libc_free_0((unsigned __int64)a1);
}
