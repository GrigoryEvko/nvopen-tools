// Function: sub_2356F30
// Address: 0x2356f30
//
void __fastcall sub_2356F30(unsigned __int64 *a1, __int64 *a2)
{
  unsigned __int64 v2; // r13
  unsigned __int64 v3; // r12
  unsigned __int64 *v4; // r15
  __int64 v5; // rdx
  char v6; // bl
  __int64 v7; // rax
  __int64 v8; // rsi
  unsigned __int64 v9; // r8
  unsigned __int64 v10; // r9
  __int64 v11; // rcx
  unsigned __int64 v12; // rdi
  unsigned __int64 v13; // rdi
  unsigned __int64 v14; // rdi
  unsigned __int64 v15; // r8
  __int64 v16; // r14
  __int64 v17; // r14
  __int64 v18; // rbx
  _QWORD *v19; // rdi
  __int64 v20; // r14
  unsigned __int64 v21; // r8
  __int64 v22; // r14
  __int64 v23; // rbx
  _QWORD *v24; // rdi
  __int64 v25; // [rsp+8h] [rbp-48h]
  __int64 v26[7]; // [rsp+18h] [rbp-38h] BYREF

  v2 = a2[1];
  v3 = a2[3];
  a2[1] = 0;
  v4 = (unsigned __int64 *)a2[4];
  v5 = *a2;
  a2[3] = 0;
  a2[4] = 0;
  v6 = *((_BYTE *)a2 + 16);
  v25 = v5;
  v7 = sub_22077B0(0x30u);
  if ( v7 )
  {
    *(_QWORD *)(v7 + 32) = v3;
    *(_QWORD *)(v7 + 16) = v2;
    *(_QWORD *)(v7 + 8) = v25;
    *(_QWORD *)v7 = &unk_4A0D878;
    *(_BYTE *)(v7 + 24) = v6;
    *(_QWORD *)(v7 + 40) = v4;
    v26[0] = v7;
    sub_2356EF0(a1, (unsigned __int64 *)v26);
    sub_23501E0(v26);
  }
  else
  {
    v26[0] = 0;
    v8 = (__int64)v26;
    sub_2356EF0(a1, (unsigned __int64 *)v26);
    sub_23501E0(v26);
    if ( v4 )
    {
      if ( (unsigned __int64 *)*v4 != v4 + 2 )
        _libc_free(*v4);
      v8 = 80;
      j_j___libc_free_0((unsigned __int64)v4);
    }
    if ( v3 )
    {
      v11 = *(unsigned int *)(v3 + 376);
      if ( (_DWORD)v11 )
      {
        v8 = (__int64)sub_ED5FB0;
        sub_EDA800(v3 + 280, (char *)sub_ED5FB0, 0, v11, v9, v10);
      }
      *(_QWORD *)(v3 + 176) = 0;
      sub_B72320(v3 + 184, v8);
      v12 = *(_QWORD *)(v3 + 152);
      if ( v12 )
        j_j___libc_free_0(v12);
      sub_C7D6A0(*(_QWORD *)(v3 + 128), 16LL * *(unsigned int *)(v3 + 144), 8);
      v13 = *(_QWORD *)(v3 + 96);
      if ( v13 )
        j_j___libc_free_0(v13);
      v14 = *(_QWORD *)(v3 + 72);
      if ( v14 )
        j_j___libc_free_0(v14);
      v15 = *(_QWORD *)(v3 + 48);
      if ( *(_DWORD *)(v3 + 60) )
      {
        v16 = *(unsigned int *)(v3 + 56);
        if ( (_DWORD)v16 )
        {
          v17 = 8 * v16;
          v18 = 0;
          do
          {
            v19 = *(_QWORD **)(v15 + v18);
            if ( v19 != (_QWORD *)-8LL && v19 )
            {
              sub_C7D6A0((__int64)v19, *v19 + 9LL, 8);
              v15 = *(_QWORD *)(v3 + 48);
            }
            v18 += 8;
          }
          while ( v17 != v18 );
        }
      }
      _libc_free(v15);
      if ( *(_DWORD *)(v3 + 36) )
      {
        v20 = *(unsigned int *)(v3 + 32);
        v21 = *(_QWORD *)(v3 + 24);
        if ( (_DWORD)v20 )
        {
          v22 = 8 * v20;
          v23 = 0;
          do
          {
            v24 = *(_QWORD **)(v21 + v23);
            if ( v24 != (_QWORD *)-8LL && v24 )
            {
              sub_C7D6A0((__int64)v24, *v24 + 9LL, 8);
              v21 = *(_QWORD *)(v3 + 24);
            }
            v23 += 8;
          }
          while ( v22 != v23 );
        }
      }
      else
      {
        v21 = *(_QWORD *)(v3 + 24);
      }
      _libc_free(v21);
      j_j___libc_free_0(v3);
    }
    if ( v2 )
    {
      sub_9CD560(v2);
      j_j___libc_free_0(v2);
    }
  }
}
