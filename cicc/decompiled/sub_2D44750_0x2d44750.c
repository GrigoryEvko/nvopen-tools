// Function: sub_2D44750
// Address: 0x2d44750
//
__int64 __fastcall sub_2D44750(__int64 *a1, __int64 a2, _QWORD *a3)
{
  __int64 v5; // rdi
  __int64 v6; // rbx
  __int64 v7; // r15
  __int64 v8; // r13
  __int64 v9; // r10
  __int64 v10; // r14
  __int64 v11; // r13
  __int64 v13; // rbx
  __int64 v14; // r13
  __int64 v15; // rdx
  unsigned int v16; // esi
  __int64 v17; // rdx
  int v18; // r14d
  __int64 v19; // rbx
  __int64 v20; // r12
  __int64 v21; // rdx
  unsigned int v22; // esi
  __int64 v23; // rbx
  __int64 v24; // r13
  __int64 v25; // rdx
  unsigned int v26; // esi
  __int64 v27; // [rsp+8h] [rbp-98h]
  __int64 v28; // [rsp+8h] [rbp-98h]
  __int64 v29; // [rsp+8h] [rbp-98h]
  _QWORD v30[4]; // [rsp+10h] [rbp-90h] BYREF
  __int16 v31; // [rsp+30h] [rbp-70h]
  _BYTE v32[32]; // [rsp+40h] [rbp-60h] BYREF
  __int16 v33; // [rsp+60h] [rbp-40h]

  v5 = a1[10];
  v6 = a3[5];
  v30[0] = "shifted";
  v31 = 259;
  v7 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, _QWORD))(*(_QWORD *)v5 + 24LL))(v5, 26, a2, v6, 0);
  if ( !v7 )
  {
    v33 = 257;
    v7 = sub_B504D0(26, a2, v6, (__int64)v32, 0, 0);
    (*(void (__fastcall **)(__int64, __int64, _QWORD *, __int64, __int64))(*(_QWORD *)a1[11] + 16LL))(
      a1[11],
      v7,
      v30,
      a1[7],
      a1[8]);
    v13 = *a1;
    v14 = *a1 + 16LL * *((unsigned int *)a1 + 2);
    if ( *a1 != v14 )
    {
      do
      {
        v15 = *(_QWORD *)(v13 + 8);
        v16 = *(_DWORD *)v13;
        v13 += 16;
        sub_B99FD0(v7, v16, v15);
      }
      while ( v14 != v13 );
    }
  }
  v8 = a3[2];
  v30[0] = "extracted";
  v31 = 259;
  if ( v8 == *(_QWORD *)(v7 + 8) )
  {
    v9 = v7;
  }
  else
  {
    v9 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)a1[10] + 120LL))(a1[10], 38, v7, v8);
    if ( !v9 )
    {
      v33 = 257;
      v28 = sub_B51D30(38, v7, v8, (__int64)v32, 0, 0);
      (*(void (__fastcall **)(__int64, __int64, _QWORD *, __int64, __int64))(*(_QWORD *)a1[11] + 16LL))(
        a1[11],
        v28,
        v30,
        a1[7],
        a1[8]);
      v23 = *a1;
      v9 = v28;
      v24 = *a1 + 16LL * *((unsigned int *)a1 + 2);
      if ( *a1 != v24 )
      {
        do
        {
          v25 = *(_QWORD *)(v23 + 8);
          v26 = *(_DWORD *)v23;
          v23 += 16;
          v29 = v9;
          sub_B99FD0(v9, v26, v25);
          v9 = v29;
        }
        while ( v24 != v23 );
      }
    }
  }
  v10 = a3[1];
  v31 = 257;
  if ( v10 == *(_QWORD *)(v9 + 8) )
    return v9;
  v27 = v9;
  v11 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)a1[10] + 120LL))(a1[10], 49, v9, v10);
  if ( !v11 )
  {
    v33 = 257;
    v11 = sub_B51D30(49, v27, v10, (__int64)v32, 0, 0);
    if ( (unsigned __int8)sub_920620(v11) )
    {
      v17 = a1[12];
      v18 = *((_DWORD *)a1 + 26);
      if ( v17 )
        sub_B99FD0(v11, 3u, v17);
      sub_B45150(v11, v18);
    }
    (*(void (__fastcall **)(__int64, __int64, _QWORD *, __int64, __int64))(*(_QWORD *)a1[11] + 16LL))(
      a1[11],
      v11,
      v30,
      a1[7],
      a1[8]);
    v19 = *a1;
    v20 = *a1 + 16LL * *((unsigned int *)a1 + 2);
    while ( v20 != v19 )
    {
      v21 = *(_QWORD *)(v19 + 8);
      v22 = *(_DWORD *)v19;
      v19 += 16;
      sub_B99FD0(v11, v22, v21);
    }
  }
  return v11;
}
