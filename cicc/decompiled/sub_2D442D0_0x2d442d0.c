// Function: sub_2D442D0
// Address: 0x2d442d0
//
__int64 __fastcall sub_2D442D0(__int64 *a1, __int64 a2, __int64 a3, __int64 *a4)
{
  __int64 v6; // r15
  __int64 v7; // r12
  __int64 v8; // r13
  __int64 v9; // r14
  __int64 v10; // rdi
  __int64 v11; // r15
  unsigned __int8 *v12; // r13
  __int64 v13; // rdi
  __int64 v14; // r12
  __int64 v15; // r15
  __int64 v16; // rdi
  __int64 v17; // r12
  __int64 v19; // r12
  __int64 i; // r15
  __int64 v21; // rdx
  unsigned int v22; // esi
  __int64 v23; // r13
  __int64 v24; // rbx
  __int64 v25; // r13
  __int64 v26; // rdx
  unsigned int v27; // esi
  __int64 v28; // r12
  __int64 v29; // r14
  __int64 v30; // rdx
  unsigned int v31; // esi
  _QWORD *v32; // rax
  __int64 v33; // r13
  __int64 v34; // r15
  __int64 v35; // rdx
  unsigned int v36; // esi
  __int64 v37; // rdx
  int v38; // r13d
  __int64 v39; // r13
  __int64 v40; // r15
  __int64 v41; // rdx
  unsigned int v42; // esi
  _QWORD v45[4]; // [rsp+10h] [rbp-90h] BYREF
  __int16 v46; // [rsp+30h] [rbp-70h]
  _BYTE v47[32]; // [rsp+40h] [rbp-60h] BYREF
  __int16 v48; // [rsp+60h] [rbp-40h]

  v6 = a4[2];
  v46 = 257;
  if ( v6 == *(_QWORD *)(a3 + 8) )
  {
    v7 = a3;
  }
  else
  {
    v7 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)a1[10] + 120LL))(a1[10], 49, a3, v6);
    if ( !v7 )
    {
      v48 = 257;
      v7 = sub_B51D30(49, a3, v6, (__int64)v47, 0, 0);
      if ( (unsigned __int8)sub_920620(v7) )
      {
        v37 = a1[12];
        v38 = *((_DWORD *)a1 + 26);
        if ( v37 )
          sub_B99FD0(v7, 3u, v37);
        sub_B45150(v7, v38);
      }
      (*(void (__fastcall **)(__int64, __int64, _QWORD *, __int64, __int64))(*(_QWORD *)a1[11] + 16LL))(
        a1[11],
        v7,
        v45,
        a1[7],
        a1[8]);
      v39 = *a1;
      v40 = *a1 + 16LL * *((unsigned int *)a1 + 2);
      if ( *a1 != v40 )
      {
        do
        {
          v41 = *(_QWORD *)(v39 + 8);
          v42 = *(_DWORD *)v39;
          v39 += 16;
          sub_B99FD0(v7, v42, v41);
        }
        while ( v40 != v39 );
      }
    }
  }
  v45[0] = "extended";
  v46 = 259;
  v8 = *a4;
  if ( *a4 == *(_QWORD *)(v7 + 8) )
  {
    v9 = v7;
  }
  else
  {
    v9 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)a1[10] + 120LL))(
           a1[10],
           39,
           v7,
           *a4);
    if ( !v9 )
    {
      v48 = 257;
      v32 = sub_BD2C40(72, 1u);
      v9 = (__int64)v32;
      if ( v32 )
        sub_B515B0((__int64)v32, v7, v8, (__int64)v47, 0, 0);
      (*(void (__fastcall **)(__int64, __int64, _QWORD *, __int64, __int64))(*(_QWORD *)a1[11] + 16LL))(
        a1[11],
        v9,
        v45,
        a1[7],
        a1[8]);
      v33 = *a1;
      v34 = *a1 + 16LL * *((unsigned int *)a1 + 2);
      if ( *a1 != v34 )
      {
        do
        {
          v35 = *(_QWORD *)(v33 + 8);
          v36 = *(_DWORD *)v33;
          v33 += 16;
          sub_B99FD0(v9, v36, v35);
        }
        while ( v34 != v33 );
      }
    }
  }
  v10 = a1[10];
  v46 = 259;
  v45[0] = "shifted";
  v11 = a4[5];
  v12 = (unsigned __int8 *)(*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64, _QWORD))(*(_QWORD *)v10 + 32LL))(
                             v10,
                             25,
                             v9,
                             v11,
                             1,
                             0);
  if ( !v12 )
  {
    v48 = 257;
    v12 = (unsigned __int8 *)sub_B504D0(25, v9, v11, (__int64)v47, 0, 0);
    (*(void (__fastcall **)(__int64, unsigned __int8 *, _QWORD *, __int64, __int64))(*(_QWORD *)a1[11] + 16LL))(
      a1[11],
      v12,
      v45,
      a1[7],
      a1[8]);
    v19 = *a1 + 16LL * *((unsigned int *)a1 + 2);
    for ( i = *a1; v19 != i; i += 16 )
    {
      v21 = *(_QWORD *)(i + 8);
      v22 = *(_DWORD *)i;
      sub_B99FD0((__int64)v12, v22, v21);
    }
    sub_B447F0(v12, 1);
  }
  v13 = a1[10];
  v45[0] = "unmasked";
  v46 = 259;
  v14 = a4[7];
  v15 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v13 + 16LL))(v13, 28, a2, v14);
  if ( !v15 )
  {
    v48 = 257;
    v15 = sub_B504D0(28, a2, v14, (__int64)v47, 0, 0);
    (*(void (__fastcall **)(__int64, __int64, _QWORD *, __int64, __int64))(*(_QWORD *)a1[11] + 16LL))(
      a1[11],
      v15,
      v45,
      a1[7],
      a1[8]);
    v28 = *a1;
    v29 = *a1 + 16LL * *((unsigned int *)a1 + 2);
    if ( *a1 != v29 )
    {
      do
      {
        v30 = *(_QWORD *)(v28 + 8);
        v31 = *(_DWORD *)v28;
        v28 += 16;
        sub_B99FD0(v15, v31, v30);
      }
      while ( v29 != v28 );
    }
  }
  v16 = a1[10];
  v46 = 259;
  v45[0] = "inserted";
  v17 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, unsigned __int8 *))(*(_QWORD *)v16 + 16LL))(
          v16,
          29,
          v15,
          v12);
  if ( !v17 )
  {
    v48 = 257;
    v17 = sub_B504D0(29, v15, (__int64)v12, (__int64)v47, 0, 0);
    (*(void (__fastcall **)(__int64, __int64, _QWORD *, __int64, __int64))(*(_QWORD *)a1[11] + 16LL))(
      a1[11],
      v17,
      v45,
      a1[7],
      a1[8]);
    v23 = 16LL * *((unsigned int *)a1 + 2);
    v24 = *a1;
    v25 = v24 + v23;
    while ( v25 != v24 )
    {
      v26 = *(_QWORD *)(v24 + 8);
      v27 = *(_DWORD *)v24;
      v24 += 16;
      sub_B99FD0(v17, v27, v26);
    }
  }
  return v17;
}
