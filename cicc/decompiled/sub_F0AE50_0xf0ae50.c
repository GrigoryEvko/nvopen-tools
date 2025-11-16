// Function: sub_F0AE50
// Address: 0xf0ae50
//
__int64 __fastcall sub_F0AE50(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r14
  _BYTE *v5; // r12
  _BYTE *v6; // r15
  __int64 v7; // rax
  __int64 v8; // r10
  bool v9; // zf
  __int64 *v10; // r14
  unsigned int *v11; // rax
  __int64 *v12; // rdi
  unsigned int v13; // r15d
  __int64 v14; // rax
  __int64 v15; // r12
  __int64 v17; // rdx
  int v18; // r13d
  __int64 v19; // rbx
  __int64 v20; // r13
  __int64 v21; // rdx
  unsigned int v22; // esi
  __int64 v23; // rdx
  int v24; // r13d
  __int64 v25; // rbx
  __int64 v26; // r13
  __int64 v27; // rdx
  unsigned int v28; // esi
  __int64 v29; // [rsp+8h] [rbp-98h]
  __int64 v30; // [rsp+8h] [rbp-98h]
  __int64 v31; // [rsp+8h] [rbp-98h]
  _BYTE v32[32]; // [rsp+10h] [rbp-90h] BYREF
  __int16 v33; // [rsp+30h] [rbp-70h]
  __int64 v34; // [rsp+40h] [rbp-60h] BYREF
  unsigned int v35; // [rsp+48h] [rbp-58h]
  __int16 v36; // [rsp+60h] [rbp-40h]

  v4 = *(_QWORD *)(a3 + 8);
  v5 = **(_BYTE ***)a1;
  v6 = **(_BYTE ***)(a1 + 8);
  if ( *v5 != 68 )
  {
    if ( !(_BYTE)a2 )
    {
      v8 = sub_AD62B0(*(_QWORD *)(a3 + 8));
      goto LABEL_8;
    }
    goto LABEL_20;
  }
  if ( (_BYTE)a2 )
  {
LABEL_20:
    v8 = sub_AD6530(*(_QWORD *)(a3 + 8), a2);
    goto LABEL_8;
  }
  v35 = sub_BCB060(*(_QWORD *)(a3 + 8));
  if ( v35 > 0x40 )
  {
    sub_C43690((__int64)&v34, 1, 0);
    v4 = *(_QWORD *)(a3 + 8);
  }
  else
  {
    v34 = 1;
  }
  v7 = sub_AD6220(v4, (__int64)&v34);
  v8 = v7;
  if ( v35 > 0x40 && v34 )
  {
    v29 = v7;
    j_j___libc_free_0_0(v34);
    v8 = v29;
  }
LABEL_8:
  v9 = v5 == v6;
  v10 = *(__int64 **)(*(_QWORD *)(a1 + 16) + 32LL);
  v11 = *(unsigned int **)(a1 + 24);
  v12 = (__int64 *)v10[10];
  v33 = 257;
  v13 = *v11;
  v14 = *v12;
  if ( v9 )
  {
    v31 = v8;
    v15 = (*(__int64 (__fastcall **)(__int64 *, _QWORD, __int64, __int64))(v14 + 16))(v12, v13, a3, v8);
    if ( !v15 )
    {
      v36 = 257;
      v15 = sub_B504D0(v13, a3, v31, (__int64)&v34, 0, 0);
      if ( (unsigned __int8)sub_920620(v15) )
      {
        v23 = v10[12];
        v24 = *((_DWORD *)v10 + 26);
        if ( v23 )
          sub_B99FD0(v15, 3u, v23);
        sub_B45150(v15, v24);
      }
      (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)v10[11] + 16LL))(
        v10[11],
        v15,
        v32,
        v10[7],
        v10[8]);
      v25 = *v10;
      v26 = *v10 + 16LL * *((unsigned int *)v10 + 2);
      if ( *v10 != v26 )
      {
        do
        {
          v27 = *(_QWORD *)(v25 + 8);
          v28 = *(_DWORD *)v25;
          v25 += 16;
          sub_B99FD0(v15, v28, v27);
        }
        while ( v26 != v25 );
      }
    }
  }
  else
  {
    v30 = v8;
    v15 = (*(__int64 (__fastcall **)(__int64 *, _QWORD, __int64, __int64))(v14 + 16))(v12, v13, v8, a3);
    if ( !v15 )
    {
      v36 = 257;
      v15 = sub_B504D0(v13, v30, a3, (__int64)&v34, 0, 0);
      if ( (unsigned __int8)sub_920620(v15) )
      {
        v17 = v10[12];
        v18 = *((_DWORD *)v10 + 26);
        if ( v17 )
          sub_B99FD0(v15, 3u, v17);
        sub_B45150(v15, v18);
      }
      (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)v10[11] + 16LL))(
        v10[11],
        v15,
        v32,
        v10[7],
        v10[8]);
      v19 = *v10;
      v20 = *v10 + 16LL * *((unsigned int *)v10 + 2);
      if ( *v10 != v20 )
      {
        do
        {
          v21 = *(_QWORD *)(v19 + 8);
          v22 = *(_DWORD *)v19;
          v19 += 16;
          sub_B99FD0(v15, v22, v21);
        }
        while ( v20 != v19 );
      }
    }
  }
  return v15;
}
