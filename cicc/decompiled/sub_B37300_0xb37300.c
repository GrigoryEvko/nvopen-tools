// Function: sub_B37300
// Address: 0xb37300
//
__int64 __fastcall sub_B37300(unsigned int **a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v7; // rax
  __int64 v8; // rdi
  __int64 v9; // rax
  unsigned int *v10; // rdi
  __int64 v11; // r12
  __int64 v12; // r14
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // r15
  __int64 v16; // r12
  unsigned int *v17; // rbx
  unsigned int *v18; // r12
  __int64 v19; // rdx
  __int64 v20; // rsi
  __int64 v22; // rcx
  int v23; // r14d
  int v24; // r15d
  _BYTE *v25; // rax
  __int64 v26; // r8
  int v27; // r14d
  unsigned int *v28; // rdi
  __int64 v29; // rax
  __int64 v30; // r12
  unsigned int *v31; // rbx
  unsigned int *v32; // r12
  __int64 v33; // rdx
  int v35; // [rsp+0h] [rbp-D0h]
  char v37[32]; // [rsp+10h] [rbp-C0h] BYREF
  __int16 v38; // [rsp+30h] [rbp-A0h]
  _QWORD v39[4]; // [rsp+40h] [rbp-90h] BYREF
  __int16 v40; // [rsp+60h] [rbp-70h]
  _BYTE *v41; // [rsp+70h] [rbp-60h] BYREF
  __int64 v42; // [rsp+78h] [rbp-58h]
  _BYTE v43[16]; // [rsp+80h] [rbp-50h] BYREF
  __int16 v44; // [rsp+90h] [rbp-40h]

  v7 = *(_QWORD *)(a2 + 8);
  if ( *(_BYTE *)(v7 + 8) == 18 )
  {
    v8 = *(_QWORD *)(*((_QWORD *)a1[6] + 9) + 40LL);
    v41 = *(_BYTE **)(a2 + 8);
    v9 = sub_B6E160(v8, 403, &v41, 1);
    v10 = a1[9];
    v39[0] = a2;
    v11 = 0;
    v12 = v9;
    v39[1] = a3;
    v13 = sub_BCB2D0(v10);
    v39[2] = sub_ACD640(v13, (unsigned int)a4, 0);
    v44 = 257;
    if ( v12 )
      v11 = *(_QWORD *)(v12 + 24);
    v14 = sub_BD2CC0(88, 4);
    v15 = v14;
    if ( v14 )
    {
      sub_B44260(v14, **(_QWORD **)(v11 + 16), 56, 4, 0, 0);
      *(_QWORD *)(v15 + 72) = 0;
      sub_B4A290(v15, v11, v12, (unsigned int)v39, 3, (unsigned int)&v41, 0, 0);
    }
    (*(void (__fastcall **)(unsigned int *, __int64, __int64, unsigned int *, unsigned int *))(*(_QWORD *)a1[11] + 16LL))(
      a1[11],
      v15,
      a5,
      a1[7],
      a1[8]);
    v16 = 4LL * *((unsigned int *)a1 + 2);
    v17 = *a1;
    v18 = &v17[v16];
    while ( v18 != v17 )
    {
      v19 = *((_QWORD *)v17 + 1);
      v20 = *v17;
      v17 += 4;
      sub_B99FD0(v15, v20, v19);
    }
  }
  else
  {
    v22 = *(unsigned int *)(v7 + 32);
    v41 = v43;
    v42 = 0x800000000LL;
    v23 = (v22 + a4) % v22;
    v24 = v22 + v23 - 1;
    v25 = v43;
    v26 = 0;
    while ( 1 )
    {
      *(_DWORD *)&v25[4 * v26] = v23;
      v26 = (unsigned int)(v42 + 1);
      LODWORD(v42) = v42 + 1;
      if ( v24 == v23 )
        break;
      if ( v26 + 1 > (unsigned __int64)HIDWORD(v42) )
      {
        sub_C8D5F0(&v41, v43, v26 + 1, 4);
        v26 = (unsigned int)v42;
      }
      v25 = v41;
      ++v23;
    }
    v27 = v26;
    v28 = a1[10];
    v35 = (int)v41;
    v38 = 257;
    v15 = (*(__int64 (__fastcall **)(unsigned int *, __int64, __int64, _BYTE *, _QWORD))(*(_QWORD *)v28 + 112LL))(
            v28,
            a2,
            a3,
            v41,
            (unsigned int)v26);
    if ( !v15 )
    {
      v40 = 257;
      v29 = sub_BD2C40(112, unk_3F1FE60);
      v15 = v29;
      if ( v29 )
        sub_B4E9E0(v29, a2, a3, v35, v27, (unsigned int)v39, 0, 0);
      a2 = v15;
      (*(void (__fastcall **)(unsigned int *, __int64, char *, unsigned int *, unsigned int *))(*(_QWORD *)a1[11] + 16LL))(
        a1[11],
        v15,
        v37,
        a1[7],
        a1[8]);
      v30 = 4LL * *((unsigned int *)a1 + 2);
      v31 = *a1;
      v32 = &v31[v30];
      while ( v32 != v31 )
      {
        v33 = *((_QWORD *)v31 + 1);
        a2 = *v31;
        v31 += 4;
        sub_B99FD0(v15, a2, v33);
      }
    }
    if ( v41 != v43 )
      _libc_free(v41, a2);
  }
  return v15;
}
