// Function: sub_B36570
// Address: 0xb36570
//
__int64 __fastcall sub_B36570(unsigned int **a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned int *v8; // rdi
  __int64 v9; // rax
  __int64 v10; // r14
  __int64 v11; // r12
  __int64 v12; // rax
  __int64 v13; // r15
  __int64 v14; // r14
  unsigned int *v15; // rdi
  __int64 v16; // r15
  __int64 v17; // r13
  __int64 v18; // r12
  __int64 v20; // r13
  unsigned int *v21; // rbx
  __int64 v22; // rdx
  __int64 v23; // rsi
  unsigned int *v24; // r12
  __int64 v25; // r13
  __int64 v26; // rdx
  __int64 v27; // rsi
  unsigned int *v28; // rdx
  unsigned int v29; // r13d
  unsigned int *v30; // r15
  __int64 v31; // r13
  __int64 v32; // rdx
  __int64 v33; // rsi
  unsigned int *v34; // rdx
  unsigned int v35; // r15d
  __int64 v36; // r15
  unsigned int *v37; // r14
  __int64 v38; // rdx
  __int64 v39; // rsi
  _BYTE v42[32]; // [rsp+10h] [rbp-90h] BYREF
  __int16 v43; // [rsp+30h] [rbp-70h]
  _BYTE v44[32]; // [rsp+40h] [rbp-60h] BYREF
  __int16 v45; // [rsp+60h] [rbp-40h]

  v8 = a1[9];
  v43 = 257;
  v9 = sub_BCB2E0(v8);
  if ( v9 == *(_QWORD *)(a3 + 8) )
  {
    v11 = a3;
  }
  else
  {
    v10 = v9;
    v11 = (*(__int64 (__fastcall **)(unsigned int *, __int64, __int64, __int64))(*(_QWORD *)a1[10] + 120LL))(
            a1[10],
            47,
            a3,
            v9);
    if ( !v11 )
    {
      v45 = 257;
      v11 = sub_B51D30(47, a3, v10, v44, 0, 0);
      if ( (unsigned __int8)sub_920620(v11) )
      {
        v34 = a1[12];
        v35 = *((_DWORD *)a1 + 26);
        if ( v34 )
          sub_B99FD0(v11, 3, v34);
        sub_B45150(v11, v35);
      }
      (*(void (__fastcall **)(unsigned int *, __int64, _BYTE *, unsigned int *, unsigned int *))(*(_QWORD *)a1[11] + 16LL))(
        a1[11],
        v11,
        v42,
        a1[7],
        a1[8]);
      v36 = (__int64)&(*a1)[4 * *((unsigned int *)a1 + 2)];
      if ( *a1 != (unsigned int *)v36 )
      {
        v37 = *a1;
        do
        {
          v38 = *((_QWORD *)v37 + 1);
          v39 = *v37;
          v37 += 4;
          sub_B99FD0(v11, v39, v38);
        }
        while ( (unsigned int *)v36 != v37 );
      }
    }
  }
  v43 = 257;
  v12 = sub_BCB2E0(a1[9]);
  v13 = v12;
  if ( v12 == *(_QWORD *)(a4 + 8) )
  {
    v14 = a4;
  }
  else
  {
    v14 = (*(__int64 (__fastcall **)(unsigned int *, __int64, __int64, __int64))(*(_QWORD *)a1[10] + 120LL))(
            a1[10],
            47,
            a4,
            v12);
    if ( !v14 )
    {
      v45 = 257;
      v14 = sub_B51D30(47, a4, v13, v44, 0, 0);
      if ( (unsigned __int8)sub_920620(v14) )
      {
        v28 = a1[12];
        v29 = *((_DWORD *)a1 + 26);
        if ( v28 )
          sub_B99FD0(v14, 3, v28);
        sub_B45150(v14, v29);
      }
      (*(void (__fastcall **)(unsigned int *, __int64, _BYTE *, unsigned int *, unsigned int *))(*(_QWORD *)a1[11] + 16LL))(
        a1[11],
        v14,
        v42,
        a1[7],
        a1[8]);
      v30 = *a1;
      v31 = (__int64)&(*a1)[4 * *((unsigned int *)a1 + 2)];
      if ( *a1 != (unsigned int *)v31 )
      {
        do
        {
          v32 = *((_QWORD *)v30 + 1);
          v33 = *v30;
          v30 += 4;
          sub_B99FD0(v14, v33, v32);
        }
        while ( (unsigned int *)v31 != v30 );
      }
    }
  }
  v15 = a1[10];
  v43 = 257;
  v16 = (*(__int64 (__fastcall **)(unsigned int *, __int64, __int64, __int64, _QWORD, _QWORD))(*(_QWORD *)v15 + 32LL))(
          v15,
          15,
          v11,
          v14,
          0,
          0);
  if ( !v16 )
  {
    v45 = 257;
    v16 = sub_B504D0(15, v11, v14, v44, 0, 0);
    (*(void (__fastcall **)(unsigned int *, __int64, _BYTE *, unsigned int *, unsigned int *))(*(_QWORD *)a1[11] + 16LL))(
      a1[11],
      v16,
      v42,
      a1[7],
      a1[8]);
    v24 = *a1;
    v25 = (__int64)&(*a1)[4 * *((unsigned int *)a1 + 2)];
    if ( *a1 != (unsigned int *)v25 )
    {
      do
      {
        v26 = *((_QWORD *)v24 + 1);
        v27 = *v24;
        v24 += 4;
        sub_B99FD0(v16, v27, v26);
      }
      while ( (unsigned int *)v25 != v24 );
    }
  }
  v17 = sub_ADB0C0(a2);
  v18 = (*(__int64 (__fastcall **)(unsigned int *, __int64, __int64, __int64, __int64))(*(_QWORD *)a1[10] + 24LL))(
          a1[10],
          20,
          v16,
          v17,
          1);
  if ( !v18 )
  {
    v45 = 257;
    v18 = sub_B504D0(20, v16, v17, v44, 0, 0);
    sub_B448B0(v18, 1);
    (*(void (__fastcall **)(unsigned int *, __int64, __int64, unsigned int *, unsigned int *))(*(_QWORD *)a1[11] + 16LL))(
      a1[11],
      v18,
      a5,
      a1[7],
      a1[8]);
    v20 = (__int64)&(*a1)[4 * *((unsigned int *)a1 + 2)];
    if ( *a1 != (unsigned int *)v20 )
    {
      v21 = *a1;
      do
      {
        v22 = *((_QWORD *)v21 + 1);
        v23 = *v21;
        v21 += 4;
        sub_B99FD0(v18, v23, v22);
      }
      while ( (unsigned int *)v20 != v21 );
    }
  }
  return v18;
}
