// Function: sub_31D8060
// Address: 0x31d8060
//
void __fastcall sub_31D8060(unsigned __int64 a1)
{
  __int64 v2; // rax
  _QWORD *v3; // rbx
  _QWORD *v4; // r12
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rbx
  _QWORD *v8; // r12
  __int64 v9; // rax
  _QWORD *v10; // rbx
  __int64 v11; // rdx
  unsigned __int64 v12; // rdi
  __int64 v13; // rbx
  _QWORD *v14; // r12
  __int64 v15; // rax
  _QWORD *v16; // rbx
  __int64 v17; // rdx
  __int64 v18; // rax
  unsigned __int64 *v19; // rax
  unsigned __int64 v20; // r14
  __int64 v21; // [rsp+0h] [rbp-60h] BYREF
  __int64 v22; // [rsp+8h] [rbp-58h]
  __int64 v23; // [rsp+10h] [rbp-50h]
  __int64 v24; // [rsp+20h] [rbp-40h] BYREF
  __int64 v25; // [rsp+28h] [rbp-38h]
  __int64 v26; // [rsp+30h] [rbp-30h]

  v2 = *(unsigned int *)(a1 + 88);
  if ( (_DWORD)v2 )
  {
    v7 = 3 * v2;
    v8 = *(_QWORD **)(a1 + 72);
    v21 = 0;
    v9 = -4096;
    v22 = 0;
    v23 = -4096;
    v10 = &v8[2 * v7];
    v24 = 0;
    v25 = 0;
    v26 = -8192;
    while ( 1 )
    {
      v11 = v8[2];
      if ( v11 != v9 )
      {
        v9 = v26;
        if ( v11 != v26 )
        {
          v12 = v8[3];
          v9 = v8[2];
          if ( v12 )
          {
            j_j___libc_free_0(v12);
            v9 = v8[2];
          }
        }
      }
      if ( v9 != 0 && v9 != -4096 && v9 != -8192 )
        sub_BD60C0(v8);
      v8 += 6;
      if ( v10 == v8 )
        break;
      v9 = v23;
    }
    if ( v23 != -4096 && v23 != 0 )
      sub_BD60C0(&v21);
    v2 = *(unsigned int *)(a1 + 88);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 72), 48 * v2, 8);
  v3 = *(_QWORD **)(a1 + 48);
  v4 = *(_QWORD **)(a1 + 40);
  if ( v3 != v4 )
  {
    do
    {
      v5 = v4[3];
      *v4 = &unk_49DB368;
      if ( v5 != -4096 && v5 != 0 && v5 != -8192 )
        sub_BD60C0(v4 + 1);
      v4 += 5;
    }
    while ( v3 != v4 );
    v4 = *(_QWORD **)(a1 + 40);
  }
  if ( v4 )
    j_j___libc_free_0((unsigned __int64)v4);
  v6 = *(unsigned int *)(a1 + 32);
  if ( (_DWORD)v6 )
  {
    v13 = 3 * v6;
    v14 = *(_QWORD **)(a1 + 16);
    v21 = 0;
    v15 = -4096;
    v22 = 0;
    v23 = -4096;
    v16 = &v14[2 * v13];
    v24 = 0;
    v25 = 0;
    v26 = -8192;
    while ( 1 )
    {
      v17 = v14[2];
      if ( v17 != v15 )
      {
        v15 = v26;
        if ( v17 != v26 )
        {
          v18 = v14[3];
          if ( v18 )
          {
            if ( (v18 & 4) != 0 )
            {
              v19 = (unsigned __int64 *)(v18 & 0xFFFFFFFFFFFFFFF8LL);
              v20 = (unsigned __int64)v19;
              if ( v19 )
              {
                if ( (unsigned __int64 *)*v19 != v19 + 2 )
                  _libc_free(*v19);
                j_j___libc_free_0(v20);
                v17 = v14[2];
              }
            }
          }
          v15 = v17;
        }
      }
      if ( v15 != -4096 && v15 != 0 && v15 != -8192 )
        sub_BD60C0(v14);
      v14 += 6;
      if ( v16 == v14 )
        break;
      v15 = v23;
    }
    if ( v26 != 0 && v26 != -4096 && v26 != -8192 )
      sub_BD60C0(&v24);
    if ( v23 != -4096 && v23 != 0 && v23 != -8192 )
      sub_BD60C0(&v21);
    v6 = *(unsigned int *)(a1 + 32);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 16), 48 * v6, 8);
  j_j___libc_free_0(a1);
}
