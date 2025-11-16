// Function: sub_B37620
// Address: 0xb37620
//
__int64 __fastcall sub_B37620(unsigned int **a1, __int64 a2, __int64 a3, __int64 *a4)
{
  __int64 v4; // r15
  __int64 **v7; // rax
  __int64 v8; // r14
  char v9; // al
  __int64 *v10; // rdx
  __int64 v11; // rax
  __int64 v12; // r15
  _DWORD *v13; // rax
  _DWORD *v14; // r13
  unsigned int v15; // r8d
  _DWORD *v16; // rdx
  char v17; // al
  __int64 v18; // rsi
  __int64 v19; // r12
  __int64 v21; // rax
  __int64 v22; // r9
  unsigned int *v23; // r14
  __int64 v24; // r13
  __int64 v25; // rdx
  __int64 v26; // rsi
  __int64 v27; // rax
  unsigned int *v28; // rbx
  __int64 v29; // r14
  __int64 v30; // rdx
  __int64 v31; // rsi
  __int64 v32; // [rsp-8h] [rbp-118h]
  __int64 v33; // [rsp+0h] [rbp-110h]
  int v34; // [rsp+10h] [rbp-100h]
  __int64 v35; // [rsp+20h] [rbp-F0h]
  unsigned int v36; // [rsp+28h] [rbp-E8h]
  _QWORD v37[4]; // [rsp+30h] [rbp-E0h] BYREF
  __int16 v38; // [rsp+50h] [rbp-C0h]
  _QWORD v39[4]; // [rsp+60h] [rbp-B0h] BYREF
  __int16 v40; // [rsp+80h] [rbp-90h]
  _DWORD *v41; // [rsp+90h] [rbp-80h] BYREF
  __int64 v42; // [rsp+98h] [rbp-78h]
  _DWORD v43[4]; // [rsp+A0h] [rbp-70h] BYREF
  __int16 v44; // [rsp+B0h] [rbp-60h]

  v36 = a2;
  v7 = (__int64 **)sub_BCE1B0(*(_QWORD *)(a3 + 8), a2);
  v8 = sub_ACADE0(v7);
  v9 = *((_BYTE *)a4 + 32);
  if ( v9 )
  {
    if ( v9 == 1 )
    {
      v39[0] = ".splatinsert";
      v40 = 259;
    }
    else
    {
      if ( *((_BYTE *)a4 + 33) == 1 )
      {
        v4 = a4[1];
        v10 = (__int64 *)*a4;
      }
      else
      {
        v10 = a4;
        v9 = 2;
      }
      v39[0] = v10;
      v39[1] = v4;
      v39[2] = ".splatinsert";
      LOBYTE(v40) = v9;
      HIBYTE(v40) = 3;
    }
  }
  else
  {
    v40 = 256;
  }
  v11 = sub_BCB2E0(a1[9]);
  v34 = sub_ACD640(v11, 0, 0);
  v12 = (*(__int64 (__fastcall **)(unsigned int *, __int64, __int64))(*(_QWORD *)a1[10] + 104LL))(a1[10], v8, a3);
  if ( !v12 )
  {
    v44 = 257;
    v21 = sub_BD2C40(72, 3);
    v12 = v21;
    if ( v21 )
    {
      sub_B4DFA0(v21, v8, a3, v34, (unsigned int)&v41, 0, 0, 0);
      v22 = v32;
    }
    (*(void (__fastcall **)(unsigned int *, __int64, _QWORD *, unsigned int *, unsigned int *, __int64))(*(_QWORD *)a1[11] + 16LL))(
      a1[11],
      v12,
      v39,
      a1[7],
      a1[8],
      v22);
    v23 = *a1;
    v24 = (__int64)&(*a1)[4 * *((unsigned int *)a1 + 2)];
    if ( *a1 != (unsigned int *)v24 )
    {
      do
      {
        v25 = *((_QWORD *)v23 + 1);
        v26 = *v23;
        v23 += 4;
        sub_B99FD0(v12, v26, v25);
      }
      while ( (unsigned int *)v24 != v23 );
    }
  }
  v13 = v43;
  v14 = v43;
  v41 = v43;
  v15 = v36;
  v42 = 0x1000000000LL;
  if ( v36 )
  {
    if ( v36 > 0x10uLL )
    {
      sub_C8D5F0(&v41, v43, v36, 4);
      v14 = v41;
      v15 = v36;
      v13 = &v41[(unsigned int)v42];
      v16 = &v41[v36];
      if ( v13 != v16 )
        goto LABEL_11;
    }
    else
    {
      v16 = &v43[v36];
      if ( v43 != v16 )
      {
        do
        {
LABEL_11:
          if ( v13 )
            *v13 = 0;
          ++v13;
        }
        while ( v16 != v13 );
        v14 = v41;
      }
    }
    LODWORD(v42) = v15;
  }
  v17 = *((_BYTE *)a4 + 32);
  if ( v17 )
  {
    if ( v17 == 1 )
    {
      v37[0] = ".splat";
      v38 = 259;
    }
    else
    {
      if ( *((_BYTE *)a4 + 33) == 1 )
      {
        v31 = a4[1];
        a4 = (__int64 *)*a4;
        v33 = v31;
      }
      else
      {
        v17 = 2;
      }
      v37[0] = a4;
      LOBYTE(v38) = v17;
      v37[1] = v33;
      v37[2] = ".splat";
      HIBYTE(v38) = 3;
    }
  }
  else
  {
    v38 = 256;
  }
  v18 = v12;
  v35 = sub_ACADE0(*(__int64 ***)(v12 + 8));
  v19 = (*(__int64 (__fastcall **)(unsigned int *, __int64, __int64, _DWORD *, _QWORD))(*(_QWORD *)a1[10] + 112LL))(
          a1[10],
          v12,
          v35,
          v14,
          v36);
  if ( !v19 )
  {
    v40 = 257;
    v27 = sub_BD2C40(112, unk_3F1FE60);
    v19 = v27;
    if ( v27 )
      sub_B4E9E0(v27, v12, v35, (_DWORD)v14, v36, (unsigned int)v39, 0, 0);
    v18 = v19;
    (*(void (__fastcall **)(unsigned int *, __int64, _QWORD *, unsigned int *, unsigned int *))(*(_QWORD *)a1[11] + 16LL))(
      a1[11],
      v19,
      v37,
      a1[7],
      a1[8]);
    v28 = *a1;
    v29 = (__int64)&(*a1)[4 * *((unsigned int *)a1 + 2)];
    if ( *a1 != (unsigned int *)v29 )
    {
      do
      {
        v30 = *((_QWORD *)v28 + 1);
        v18 = *v28;
        v28 += 4;
        sub_B99FD0(v19, v18, v30);
      }
      while ( (unsigned int *)v29 != v28 );
    }
  }
  if ( v41 != v43 )
    _libc_free(v41, v18);
  return v19;
}
