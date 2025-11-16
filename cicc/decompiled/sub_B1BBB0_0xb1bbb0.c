// Function: sub_B1BBB0
// Address: 0xb1bbb0
//
__int64 __fastcall sub_B1BBB0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *v5; // rax
  __int64 v6; // r12
  int v7; // eax
  __int64 v8; // rdx
  __int64 v9; // r8
  unsigned __int64 v10; // rcx
  __int64 v11; // rdx
  __int64 v12; // r13
  __int64 v13; // rdi
  __int64 v15; // rax
  unsigned int v16; // edx
  unsigned int v17; // r15d
  __int64 v18; // r13
  _QWORD *v19; // rax
  _QWORD *v20; // r13
  __int64 v21; // rax
  __int64 v22; // r13
  __int64 v23; // r9
  __int64 v24; // rdi
  __int64 v25; // [rsp+8h] [rbp-48h]
  __int64 v26; // [rsp+10h] [rbp-40h]
  __int64 v27; // [rsp+10h] [rbp-40h]
  __int64 v28; // [rsp+18h] [rbp-38h]
  __int64 v29; // [rsp+18h] [rbp-38h]
  __int64 v30; // [rsp+18h] [rbp-38h]

  v5 = (__int64 *)sub_22077B0(80);
  v6 = (__int64)v5;
  if ( v5 )
  {
    *v5 = a2;
    v5[1] = a3;
    v7 = 0;
    if ( a3 )
      v7 = *(_DWORD *)(a3 + 16) + 1;
    *(_DWORD *)(v6 + 16) = v7;
    *(_QWORD *)(v6 + 24) = v6 + 40;
    *(_QWORD *)(v6 + 32) = 0x400000000LL;
    *(_QWORD *)(v6 + 72) = -1;
  }
  if ( a2 )
  {
    v8 = (unsigned int)(*(_DWORD *)(a2 + 44) + 1);
    v9 = 8 * v8;
  }
  else
  {
    v9 = 0;
    LODWORD(v8) = 0;
  }
  v10 = *(unsigned int *)(a1 + 56);
  if ( (unsigned int)v10 > (unsigned int)v8 )
    goto LABEL_8;
  v15 = *(_QWORD *)(a1 + 128);
  v16 = v8 + 1;
  if ( *(_DWORD *)(v15 + 88) >= v16 )
    v16 = *(_DWORD *)(v15 + 88);
  a2 = v16;
  v17 = v16;
  if ( v16 == v10 )
  {
LABEL_8:
    v11 = *(_QWORD *)(a1 + 48);
    goto LABEL_9;
  }
  v18 = 8LL * v16;
  if ( v16 < v10 )
  {
    v11 = *(_QWORD *)(a1 + 48);
    v21 = v11 + 8 * v10;
    v22 = v11 + v18;
    if ( v21 == v22 )
      goto LABEL_27;
    do
    {
      v23 = *(_QWORD *)(v21 - 8);
      v21 -= 8;
      if ( v23 )
      {
        v24 = *(_QWORD *)(v23 + 24);
        if ( v24 != v23 + 40 )
        {
          v25 = v23;
          v26 = v21;
          v29 = v9;
          _libc_free(v24, a2);
          v23 = v25;
          v21 = v26;
          v9 = v29;
        }
        a2 = 80;
        v27 = v21;
        v30 = v9;
        j_j___libc_free_0(v23, 80);
        v21 = v27;
        v9 = v30;
      }
    }
    while ( v22 != v21 );
  }
  else
  {
    if ( v16 > (unsigned __int64)*(unsigned int *)(a1 + 60) )
    {
      v28 = v9;
      sub_B1B4E0(a1 + 48, v16);
      v10 = *(unsigned int *)(a1 + 56);
      v9 = v28;
    }
    v11 = *(_QWORD *)(a1 + 48);
    v19 = (_QWORD *)(v11 + 8 * v10);
    v20 = (_QWORD *)(v11 + v18);
    if ( v19 == v20 )
      goto LABEL_27;
    do
    {
      if ( v19 )
        *v19 = 0;
      ++v19;
    }
    while ( v20 != v19 );
  }
  v11 = *(_QWORD *)(a1 + 48);
LABEL_27:
  *(_DWORD *)(a1 + 56) = v17;
LABEL_9:
  v12 = *(_QWORD *)(v11 + v9);
  *(_QWORD *)(v11 + v9) = v6;
  if ( v12 )
  {
    v13 = *(_QWORD *)(v12 + 24);
    if ( v13 != v12 + 40 )
      _libc_free(v13, a2);
    j_j___libc_free_0(v12, 80);
  }
  if ( a3 )
    sub_B1AE00(a3 + 24, v6);
  return v6;
}
