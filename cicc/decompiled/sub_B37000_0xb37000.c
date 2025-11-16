// Function: sub_B37000
// Address: 0xb37000
//
__int64 __fastcall sub_B37000(unsigned int **a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  int v5; // r14d
  int v6; // r14d
  _BYTE *v7; // rax
  __int64 v8; // r8
  __int64 v9; // r13
  __int64 v10; // r15
  __int64 v11; // rsi
  __int64 v12; // r14
  __int64 v14; // rax
  unsigned int *v15; // rbx
  __int64 v16; // r13
  __int64 v17; // rdx
  __int64 v18; // r15
  __int64 v19; // rdi
  __int64 v20; // rax
  int v21; // r12d
  __int64 v22; // rax
  __int64 v23; // r12
  unsigned int *v24; // rbx
  unsigned int *v25; // r12
  __int64 v26; // rdx
  __int64 v27; // rsi
  __int64 v29; // [rsp+8h] [rbp-A8h]
  _BYTE *v30; // [rsp+10h] [rbp-A0h]
  __int64 v31; // [rsp+18h] [rbp-98h] BYREF
  char v32; // [rsp+20h] [rbp-90h] BYREF
  __int16 v33; // [rsp+40h] [rbp-70h]
  _BYTE *v34; // [rsp+50h] [rbp-60h] BYREF
  __int64 v35; // [rsp+58h] [rbp-58h]
  _BYTE v36[16]; // [rsp+60h] [rbp-50h] BYREF
  __int16 v37; // [rsp+70h] [rbp-40h]

  v4 = *(_QWORD *)(a2 + 8);
  v31 = a2;
  if ( *(_BYTE *)(v4 + 8) == 18 )
  {
    v18 = 0;
    v19 = *(_QWORD *)(*((_QWORD *)a1[6] + 9) + 40LL);
    v34 = (_BYTE *)v4;
    v20 = sub_B6E160(v19, 402, &v34, 1);
    v37 = 257;
    v21 = v20;
    if ( v20 )
      v18 = *(_QWORD *)(v20 + 24);
    v22 = sub_BD2CC0(88, 2);
    v12 = v22;
    if ( v22 )
    {
      sub_B44260(v22, **(_QWORD **)(v18 + 16), 56, 2, 0, 0);
      *(_QWORD *)(v12 + 72) = 0;
      sub_B4A290(v12, v18, v21, (unsigned int)&v31, 1, (unsigned int)&v34, 0, 0);
    }
    (*(void (__fastcall **)(unsigned int *, __int64, __int64, unsigned int *, unsigned int *))(*(_QWORD *)a1[11] + 16LL))(
      a1[11],
      v12,
      a3,
      a1[7],
      a1[8]);
    v23 = 4LL * *((unsigned int *)a1 + 2);
    v24 = *a1;
    v25 = &v24[v23];
    while ( v25 != v24 )
    {
      v26 = *((_QWORD *)v24 + 1);
      v27 = *v24;
      v24 += 4;
      sub_B99FD0(v12, v27, v26);
    }
  }
  else
  {
    v34 = v36;
    v35 = 0x800000000LL;
    v5 = *(_DWORD *)(v4 + 32);
    if ( v5 <= 0 )
    {
      v30 = v36;
      v9 = a2;
      v10 = 0;
    }
    else
    {
      v6 = v5 - 1;
      v7 = v36;
      v8 = 0;
      while ( 1 )
      {
        *(_DWORD *)&v7[4 * v8] = v6;
        v8 = (unsigned int)(v35 + 1);
        LODWORD(v35) = v35 + 1;
        if ( !v6 )
          break;
        if ( v8 + 1 > (unsigned __int64)HIDWORD(v35) )
        {
          sub_C8D5F0(&v34, v36, v8 + 1, 4);
          v8 = (unsigned int)v35;
        }
        v7 = v34;
        --v6;
      }
      v9 = v31;
      v10 = (unsigned int)v8;
      v30 = v34;
    }
    v11 = v9;
    v29 = sub_ACADE0(*(__int64 ***)(v9 + 8));
    v12 = (*(__int64 (__fastcall **)(unsigned int *, __int64, __int64, _BYTE *, __int64))(*(_QWORD *)a1[10] + 112LL))(
            a1[10],
            v9,
            v29,
            v30,
            v10);
    if ( !v12 )
    {
      v33 = 257;
      v14 = sub_BD2C40(112, unk_3F1FE60);
      v12 = v14;
      if ( v14 )
        sub_B4E9E0(v14, v9, v29, (_DWORD)v30, v10, (unsigned int)&v32, 0, 0);
      v11 = v12;
      (*(void (__fastcall **)(unsigned int *, __int64, __int64, unsigned int *, unsigned int *))(*(_QWORD *)a1[11] + 16LL))(
        a1[11],
        v12,
        a3,
        a1[7],
        a1[8]);
      v15 = *a1;
      v16 = (__int64)&(*a1)[4 * *((unsigned int *)a1 + 2)];
      if ( *a1 != (unsigned int *)v16 )
      {
        do
        {
          v17 = *((_QWORD *)v15 + 1);
          v11 = *v15;
          v15 += 4;
          sub_B99FD0(v12, v11, v17);
        }
        while ( (unsigned int *)v16 != v15 );
      }
    }
    if ( v34 != v36 )
      _libc_free(v34, v11);
  }
  return v12;
}
