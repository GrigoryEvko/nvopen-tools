// Function: sub_EDAF70
// Address: 0xedaf70
//
__int64 __fastcall sub_EDAF70(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int64 a5, unsigned __int64 a6)
{
  __int64 v7; // rdi
  __int64 v8; // rdi
  _QWORD *v9; // r13
  _QWORD *v10; // rdi
  _QWORD *v11; // rdi
  _QWORD *v12; // rdi
  __int64 v13; // rdi
  __int64 v14; // r13
  __int64 v15; // rdi
  __int64 v16; // r13
  __int64 v17; // rdi
  __int64 v18; // rdi
  _QWORD *v19; // rdx
  __int64 (__fastcall *v20)(_QWORD *); // rax
  __int64 v21; // rax
  _QWORD *v22; // r13
  _QWORD *v23; // rax
  _QWORD *v24; // r14
  _QWORD *v25; // rdi
  _QWORD *v26; // rbx
  _QWORD *v27; // r15
  __int64 v28; // rdi
  __int64 v29; // rdi
  __int64 v30; // rdi
  __int64 v31; // rbx
  __int64 v32; // r13
  __int64 v33; // rdi
  __int64 v34; // rdi
  __int64 v35; // rdi
  __int64 result; // rax
  __int64 v37; // [rsp+0h] [rbp-50h]
  _QWORD *v38; // [rsp+8h] [rbp-48h]
  _QWORD *v39; // [rsp+10h] [rbp-40h]
  _QWORD *v40; // [rsp+18h] [rbp-38h]

  *(_QWORD *)a1 = &unk_49E4DE8;
  v7 = *(_QWORD *)(a1 + 424);
  if ( v7 )
  {
    a2 = 48;
    j_j___libc_free_0(v7, 48);
  }
  v8 = *(_QWORD *)(a1 + 416);
  if ( v8 )
  {
    a2 = 48;
    j_j___libc_free_0(v8, 48);
  }
  v9 = *(_QWORD **)(a1 + 408);
  if ( v9 )
  {
    v10 = (_QWORD *)v9[58];
    if ( v10 != v9 + 60 )
      _libc_free(v10, a2);
    v11 = (_QWORD *)v9[35];
    if ( v11 != v9 + 37 )
      _libc_free(v11, a2);
    v12 = (_QWORD *)v9[5];
    if ( v12 != v9 + 7 )
      _libc_free(v12, a2);
    a2 = 536;
    j_j___libc_free_0(v9, 536);
  }
  v13 = *(_QWORD *)(a1 + 168);
  if ( v13 != a1 + 184 )
    _libc_free(v13, a2);
  v14 = *(_QWORD *)(a1 + 152);
  if ( v14 )
  {
    v15 = *(_QWORD *)(v14 + 8);
    if ( v15 )
      j_j___libc_free_0(v15, *(_QWORD *)(v14 + 24) - v15);
    a2 = 88;
    j_j___libc_free_0(v14, 88);
  }
  v16 = *(_QWORD *)(a1 + 144);
  if ( v16 )
  {
    v17 = *(_QWORD *)(v16 + 8);
    if ( v17 )
      j_j___libc_free_0(v17, *(_QWORD *)(v16 + 24) - v17);
    a2 = 88;
    j_j___libc_free_0(v16, 88);
  }
  v18 = *(_QWORD *)(a1 + 136);
  if ( v18 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v18 + 8LL))(v18);
  v19 = *(_QWORD **)(a1 + 128);
  v38 = v19;
  if ( v19 )
  {
    v20 = *(__int64 (__fastcall **)(_QWORD *))(*v19 + 8LL);
    if ( v20 == sub_ED7520 )
    {
      *v19 = &unk_49E4D18;
      v21 = v19[1];
      v37 = v21;
      if ( v21 )
      {
        v22 = *(_QWORD **)(v21 + 32);
        v39 = *(_QWORD **)(v21 + 40);
        if ( v39 != v22 )
        {
          do
          {
            v23 = (_QWORD *)v22[6];
            v40 = v23;
            if ( v23 )
            {
              v24 = v23 + 9;
              do
              {
                v25 = (_QWORD *)*(v24 - 3);
                v26 = (_QWORD *)*(v24 - 2);
                v24 -= 3;
                v27 = v25;
                if ( v26 != v25 )
                {
                  do
                  {
                    if ( *v27 )
                      j_j___libc_free_0(*v27, v27[2] - *v27);
                    v27 += 3;
                  }
                  while ( v26 != v27 );
                  v25 = (_QWORD *)*v24;
                }
                if ( v25 )
                  j_j___libc_free_0(v25, v24[2] - (_QWORD)v25);
              }
              while ( v40 != v24 );
              j_j___libc_free_0(v40, 72);
            }
            v28 = v22[3];
            if ( v28 )
              j_j___libc_free_0(v28, v22[5] - v28);
            if ( *v22 )
              j_j___libc_free_0(*v22, v22[2] - *v22);
            v22 += 10;
          }
          while ( v39 != v22 );
          v22 = *(_QWORD **)(v37 + 32);
        }
        if ( v22 )
          j_j___libc_free_0(v22, *(_QWORD *)(v37 + 48) - (_QWORD)v22);
        j_j___libc_free_0(v37, 80);
      }
      a2 = 56;
      j_j___libc_free_0(v38, 56);
    }
    else
    {
      v20(v19);
    }
  }
  v29 = *(_QWORD *)(a1 + 120);
  if ( v29 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v29 + 8LL))(v29);
  v30 = *(_QWORD *)(a1 + 112);
  if ( v30 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v30 + 8LL))(v30);
  v31 = *(_QWORD *)(a1 + 56);
  v32 = v31 + 32LL * *(unsigned int *)(a1 + 64);
  *(_QWORD *)a1 = &unk_49E4C18;
  if ( v31 != v32 )
  {
    do
    {
      v33 = *(_QWORD *)(v32 - 32);
      v32 -= 32;
      if ( v33 )
      {
        a2 = *(_QWORD *)(v32 + 16) - v33;
        j_j___libc_free_0(v33, a2);
      }
    }
    while ( v31 != v32 );
    v32 = *(_QWORD *)(a1 + 56);
  }
  if ( v32 != a1 + 72 )
    _libc_free(v32, a2);
  v34 = *(_QWORD *)(a1 + 48);
  if ( v34 )
    sub_EDAD10(v34, a2, (__int64)v19, a4, a5, a6);
  v35 = *(_QWORD *)(a1 + 16);
  result = a1 + 32;
  if ( v35 != a1 + 32 )
    return j_j___libc_free_0(v35, *(_QWORD *)(a1 + 32) + 1LL);
  return result;
}
