// Function: sub_256B960
// Address: 0x256b960
//
void __fastcall sub_256B960(__int64 a1, __int64 a2, unsigned __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  __int64 v7; // r12
  __int64 *i; // rax
  __int64 v9; // r9
  int v10; // eax
  __int64 v11; // rdi
  __int64 v12; // rax
  int *v13; // rdx
  unsigned __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  unsigned __int64 v19; // r14
  unsigned __int64 v20; // rdi
  __int64 v21; // rdx
  int v22; // ecx
  _QWORD *v23; // rdx
  unsigned __int64 v24; // rbx
  unsigned __int64 v25; // rdi
  unsigned __int64 v26; // [rsp+8h] [rbp-108h]
  __int64 v27; // [rsp+10h] [rbp-100h]
  __int64 v28; // [rsp+10h] [rbp-100h]
  char v29; // [rsp+1Fh] [rbp-F1h]
  char *v30; // [rsp+20h] [rbp-F0h] BYREF
  __int64 v31; // [rsp+28h] [rbp-E8h]
  _BYTE v32[40]; // [rsp+30h] [rbp-E0h] BYREF
  int v33; // [rsp+58h] [rbp-B8h] BYREF
  __int64 v34; // [rsp+60h] [rbp-B0h]
  int *v35; // [rsp+68h] [rbp-A8h]
  int *v36; // [rsp+70h] [rbp-A0h]
  __int64 v37; // [rsp+78h] [rbp-98h]
  unsigned __int64 v38[2]; // [rsp+80h] [rbp-90h] BYREF
  _BYTE v39[40]; // [rsp+90h] [rbp-80h] BYREF
  int v40; // [rsp+B8h] [rbp-58h] BYREF
  __int64 v41; // [rsp+C0h] [rbp-50h]
  int *v42; // [rsp+C8h] [rbp-48h]
  int *v43; // [rsp+D0h] [rbp-40h]
  __int64 v44; // [rsp+D8h] [rbp-38h]

  v6 = *(_QWORD *)(a1 + 376);
  if ( v6 )
  {
    if ( v6 != 1 )
    {
      v33 = 0;
      v30 = v32;
      v31 = 0x400000000LL;
      v34 = 0;
      v35 = &v33;
      v36 = &v33;
      v37 = 0;
      goto LABEL_32;
    }
    a4 = *(_QWORD *)(a1 + 360);
    v23 = (_QWORD *)(a4 + 32);
LABEL_35:
    if ( *v23 == 0x7FFFFFFF )
    {
      v24 = *(_QWORD *)(a2 + 64);
      *(_DWORD *)(a2 + 8) = 0;
      while ( v24 )
      {
        sub_253B2D0(*(_QWORD *)(v24 + 24));
        v25 = v24;
        v24 = *(_QWORD *)(v24 + 16);
        j_j___libc_free_0(v25);
      }
      *(_QWORD *)(a2 + 64) = 0;
      *(_QWORD *)(a2 + 72) = a2 + 56;
      *(_QWORD *)(a2 + 80) = a2 + 56;
      *(_QWORD *)(a2 + 88) = 0;
      sub_256AFA0((__int64)v38, a2, &qword_438A698, a4, a5);
      return;
    }
    a3 = (unsigned __int64)&v33;
    v33 = 0;
    v30 = v32;
    a4 = 0x400000000LL;
    v31 = 0x400000000LL;
    v34 = 0;
    v35 = &v33;
    v36 = &v33;
    v37 = 0;
    if ( !v6 )
    {
      a3 = *(unsigned int *)(a1 + 296);
      goto LABEL_4;
    }
LABEL_32:
    v29 = 0;
    v7 = *(_QWORD *)(a1 + 360);
    v26 = a1 + 344;
    goto LABEL_5;
  }
  a3 = *(unsigned int *)(a1 + 296);
  if ( *(_DWORD *)(a1 + 296) && a3 == 1 )
  {
    v23 = *(_QWORD **)(a1 + 288);
    goto LABEL_35;
  }
  v33 = 0;
  v30 = v32;
  v31 = 0x400000000LL;
  v34 = 0;
  v35 = &v33;
  v36 = &v33;
  v37 = 0;
LABEL_4:
  v7 = *(_QWORD *)(a1 + 288);
  v29 = 1;
  v26 = v7 + 8 * a3;
LABEL_5:
  if ( v29 )
    goto LABEL_22;
  for ( ; v26 != v7; v7 = sub_220EF30(v7) )
  {
    for ( i = (__int64 *)(v7 + 32); ; i = (__int64 *)v7 )
    {
      v9 = *i;
      v38[1] = 0x400000000LL;
      v10 = *(_DWORD *)(a2 + 8);
      v38[0] = (unsigned __int64)v39;
      if ( v10 )
      {
        v28 = v9;
        sub_2538630((__int64)v38, a2, a3, a4, a5, v9);
        v9 = v28;
      }
      v11 = *(_QWORD *)(a2 + 64);
      v40 = 0;
      v41 = 0;
      v42 = &v40;
      v43 = &v40;
      v44 = 0;
      if ( v11 )
      {
        v27 = v9;
        v12 = sub_25383A0(v11, (__int64)&v40);
        v9 = v27;
        a4 = v12;
        do
        {
          v13 = (int *)v12;
          v12 = *(_QWORD *)(v12 + 16);
        }
        while ( v12 );
        v42 = v13;
        v14 = a4;
        do
        {
          a3 = v14;
          v14 = *(_QWORD *)(v14 + 24);
        }
        while ( v14 );
        v15 = *(_QWORD *)(a2 + 88);
        v43 = (int *)a3;
        v41 = a4;
        v44 = v15;
      }
      sub_256B620((__int64)v38, v9, a3, a4, a5, v9);
      sub_256B200((__int64)&v30, (__int64)v38, v16, v17, v18);
      v19 = v41;
      if ( v41 )
      {
        do
        {
          sub_253B2D0(*(_QWORD *)(v19 + 24));
          v20 = v19;
          v19 = *(_QWORD *)(v19 + 16);
          j_j___libc_free_0(v20);
        }
        while ( v19 );
      }
      if ( (_BYTE *)v38[0] != v39 )
        _libc_free(v38[0]);
      if ( !v29 )
        break;
      v7 += 8;
LABEL_22:
      if ( v26 == v7 )
        goto LABEL_25;
    }
  }
LABEL_25:
  sub_25387F0(a2, &v30, a3, a4, a5, a6);
  sub_253B2D0(*(_QWORD *)(a2 + 64));
  v21 = v34;
  *(_QWORD *)(a2 + 64) = 0;
  *(_QWORD *)(a2 + 72) = a2 + 56;
  *(_QWORD *)(a2 + 80) = a2 + 56;
  *(_QWORD *)(a2 + 88) = 0;
  if ( v21 )
  {
    v22 = v33;
    *(_QWORD *)(a2 + 64) = v21;
    *(_DWORD *)(a2 + 56) = v22;
    *(_QWORD *)(a2 + 72) = v35;
    *(_QWORD *)(a2 + 80) = v36;
    *(_QWORD *)(v21 + 8) = a2 + 56;
    *(_QWORD *)(a2 + 88) = v37;
  }
  if ( v30 != v32 )
    _libc_free((unsigned __int64)v30);
}
