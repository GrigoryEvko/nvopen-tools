// Function: sub_2A20570
// Address: 0x2a20570
//
__int64 __fastcall sub_2A20570(__int64 *a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 v6; // rsi
  __int64 v7; // rax
  __int64 v8; // r13
  __int64 v9; // r14
  __int64 v10; // r15
  __int64 v11; // rdi
  __int64 v12; // r14
  __int64 v14; // rdi
  __int64 v16; // r15
  __int64 v17; // rbx
  __int64 v18; // r12
  __int64 v19; // rdx
  unsigned int v20; // esi
  __int64 v21; // rbx
  __int64 v22; // r12
  __int64 v23; // rdx
  unsigned int v24; // esi
  __int64 v25; // rbx
  __int64 v26; // r14
  __int64 v27; // rdx
  unsigned int v28; // esi
  __int64 v29; // rbx
  __int64 v30; // r14
  __int64 v31; // rdx
  unsigned int v32; // esi
  __int64 v33; // [rsp+8h] [rbp-98h]
  _QWORD v34[4]; // [rsp+10h] [rbp-90h] BYREF
  __int16 v35; // [rsp+30h] [rbp-70h]
  _BYTE v36[32]; // [rsp+40h] [rbp-60h] BYREF
  __int16 v37; // [rsp+60h] [rbp-40h]

  if ( a4 && (v6 = a4 - 1, ((unsigned int)v6 & a4) == 0) )
  {
    v14 = *(_QWORD *)(a3 + 8);
    v35 = 259;
    v34[0] = "xtraiter";
    v16 = sub_AD64C0(v14, v6, 0);
    v12 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)a1[10] + 16LL))(
            a1[10],
            28,
            a3,
            v16);
    if ( !v12 )
    {
      v37 = 257;
      v12 = sub_B504D0(28, a3, v16, (__int64)v36, 0, 0);
      (*(void (__fastcall **)(__int64, __int64, _QWORD *, __int64, __int64))(*(_QWORD *)a1[11] + 16LL))(
        a1[11],
        v12,
        v34,
        a1[7],
        a1[8]);
      v17 = *a1;
      v18 = *a1 + 16LL * *((unsigned int *)a1 + 2);
      while ( v18 != v17 )
      {
        v19 = *(_QWORD *)(v17 + 8);
        v20 = *(_DWORD *)v17;
        v17 += 16;
        sub_B99FD0(v12, v20, v19);
      }
    }
  }
  else
  {
    v7 = sub_AD64C0(*(_QWORD *)(a2 + 8), a4, 0);
    v35 = 257;
    v33 = v7;
    v8 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)a1[10] + 16LL))(a1[10], 22, a2, v7);
    if ( !v8 )
    {
      v37 = 257;
      v8 = sub_B504D0(22, a2, v33, (__int64)v36, 0, 0);
      (*(void (__fastcall **)(__int64, __int64, _QWORD *, __int64, __int64))(*(_QWORD *)a1[11] + 16LL))(
        a1[11],
        v8,
        v34,
        a1[7],
        a1[8]);
      v29 = *a1;
      v30 = *a1 + 16LL * *((unsigned int *)a1 + 2);
      if ( *a1 != v30 )
      {
        do
        {
          v31 = *(_QWORD *)(v29 + 8);
          v32 = *(_DWORD *)v29;
          v29 += 16;
          sub_B99FD0(v8, v32, v31);
        }
        while ( v30 != v29 );
      }
    }
    v35 = 257;
    v9 = sub_AD64C0(*(_QWORD *)(v8 + 8), 1, 0);
    v10 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, _QWORD, _QWORD))(*(_QWORD *)a1[10] + 32LL))(
            a1[10],
            13,
            v8,
            v9,
            0,
            0);
    if ( !v10 )
    {
      v37 = 257;
      v10 = sub_B504D0(13, v8, v9, (__int64)v36, 0, 0);
      (*(void (__fastcall **)(__int64, __int64, _QWORD *, __int64, __int64))(*(_QWORD *)a1[11] + 16LL))(
        a1[11],
        v10,
        v34,
        a1[7],
        a1[8]);
      v25 = *a1;
      v26 = *a1 + 16LL * *((unsigned int *)a1 + 2);
      if ( *a1 != v26 )
      {
        do
        {
          v27 = *(_QWORD *)(v25 + 8);
          v28 = *(_DWORD *)v25;
          v25 += 16;
          sub_B99FD0(v10, v28, v27);
        }
        while ( v26 != v25 );
      }
    }
    v11 = a1[10];
    v35 = 259;
    v34[0] = "xtraiter";
    v12 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v11 + 16LL))(v11, 22, v10, v33);
    if ( !v12 )
    {
      v37 = 257;
      v12 = sub_B504D0(22, v10, v33, (__int64)v36, 0, 0);
      (*(void (__fastcall **)(__int64, __int64, _QWORD *, __int64, __int64))(*(_QWORD *)a1[11] + 16LL))(
        a1[11],
        v12,
        v34,
        a1[7],
        a1[8]);
      v21 = *a1;
      v22 = *a1 + 16LL * *((unsigned int *)a1 + 2);
      while ( v22 != v21 )
      {
        v23 = *(_QWORD *)(v21 + 8);
        v24 = *(_DWORD *)v21;
        v21 += 16;
        sub_B99FD0(v12, v24, v23);
      }
    }
  }
  return v12;
}
