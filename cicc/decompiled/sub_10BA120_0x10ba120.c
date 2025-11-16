// Function: sub_10BA120
// Address: 0x10ba120
//
__int64 __fastcall sub_10BA120(__int64 *a1, unsigned int a2, __int64 a3, __int64 a4)
{
  __int64 *v7; // rax
  __int64 v8; // r12
  __int64 v9; // r13
  __int64 v10; // r15
  unsigned int *v11; // rax
  __int64 v12; // r14
  __int64 v13; // r12
  __int64 v14; // r15
  __int64 v15; // rdi
  __int64 v16; // rdx
  __int64 v18; // r15
  __int64 v19; // r12
  __int64 v20; // rdx
  unsigned int v21; // esi
  __int64 v22; // r13
  __int64 v23; // r12
  __int64 v24; // r13
  __int64 v25; // rdx
  unsigned int v26; // esi
  _QWORD *v27; // rax
  __int64 v28; // r14
  __int64 v29; // r12
  __int64 v30; // rdx
  unsigned int v31; // esi
  _QWORD *v32; // rax
  __int64 v33; // r15
  __int64 v34; // r12
  __int64 v35; // rdx
  unsigned int v36; // esi
  __int64 v37; // [rsp+8h] [rbp-B8h]
  __int64 v38; // [rsp+8h] [rbp-B8h]
  __int64 v40; // [rsp+18h] [rbp-A8h]
  __int64 v41; // [rsp+28h] [rbp-98h] BYREF
  _DWORD v42[8]; // [rsp+30h] [rbp-90h] BYREF
  __int16 v43; // [rsp+50h] [rbp-70h]
  _BYTE v44[32]; // [rsp+60h] [rbp-60h] BYREF
  __int16 v45; // [rsp+80h] [rbp-40h]

  v7 = (__int64 *)a1[1];
  v43 = 257;
  v8 = *a1;
  if ( *v7 == *(_QWORD *)(a3 + 8) )
  {
    v9 = a3;
  }
  else
  {
    v40 = *v7;
    v9 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64))(**(_QWORD **)(v8 + 80) + 120LL))(
           *(_QWORD *)(v8 + 80),
           39,
           a3,
           *v7);
    if ( v9 )
    {
      v8 = *a1;
      v7 = (__int64 *)a1[1];
    }
    else
    {
      v45 = 257;
      v32 = sub_BD2C40(72, unk_3F10A14);
      v9 = (__int64)v32;
      if ( v32 )
        sub_B515B0((__int64)v32, a3, v40, (__int64)v44, 0, 0);
      (*(void (__fastcall **)(_QWORD, __int64, _DWORD *, _QWORD, _QWORD))(**(_QWORD **)(v8 + 88) + 16LL))(
        *(_QWORD *)(v8 + 88),
        v9,
        v42,
        *(_QWORD *)(v8 + 56),
        *(_QWORD *)(v8 + 64));
      v33 = *(_QWORD *)v8;
      v34 = *(_QWORD *)v8 + 16LL * *(unsigned int *)(v8 + 8);
      while ( v34 != v33 )
      {
        v35 = *(_QWORD *)(v33 + 8);
        v36 = *(_DWORD *)v33;
        v33 += 16;
        sub_B99FD0(v9, v36, v35);
      }
      v8 = *a1;
      v7 = (__int64 *)a1[1];
    }
  }
  v43 = 257;
  if ( *v7 == *(_QWORD *)(a4 + 8) )
  {
    v10 = a4;
  }
  else
  {
    v37 = *v7;
    v10 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64))(**(_QWORD **)(v8 + 80) + 120LL))(
            *(_QWORD *)(v8 + 80),
            39,
            a4,
            *v7);
    if ( !v10 )
    {
      v45 = 257;
      v27 = sub_BD2C40(72, unk_3F10A14);
      v10 = (__int64)v27;
      if ( v27 )
        sub_B515B0((__int64)v27, a4, v37, (__int64)v44, 0, 0);
      (*(void (__fastcall **)(_QWORD, __int64, _DWORD *, _QWORD, _QWORD))(**(_QWORD **)(v8 + 88) + 16LL))(
        *(_QWORD *)(v8 + 88),
        v10,
        v42,
        *(_QWORD *)(v8 + 56),
        *(_QWORD *)(v8 + 64));
      v28 = *(_QWORD *)v8;
      v29 = *(_QWORD *)v8 + 16LL * *(unsigned int *)(v8 + 8);
      while ( v29 != v28 )
      {
        v30 = *(_QWORD *)(v28 + 8);
        v31 = *(_DWORD *)v28;
        v28 += 16;
        sub_B99FD0(v10, v31, v30);
      }
    }
    v8 = *a1;
  }
  v11 = (unsigned int *)a1[2];
  v43 = 257;
  v38 = sub_AD64C0(*(_QWORD *)(v10 + 8), *v11, 0);
  v12 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64, _QWORD, _QWORD))(**(_QWORD **)(v8 + 80) + 32LL))(
          *(_QWORD *)(v8 + 80),
          25,
          v10,
          v38,
          0,
          0);
  if ( !v12 )
  {
    v45 = 257;
    v12 = sub_B504D0(25, v10, v38, (__int64)v44, 0, 0);
    (*(void (__fastcall **)(_QWORD, __int64, _DWORD *, _QWORD, _QWORD))(**(_QWORD **)(v8 + 88) + 16LL))(
      *(_QWORD *)(v8 + 88),
      v12,
      v42,
      *(_QWORD *)(v8 + 56),
      *(_QWORD *)(v8 + 64));
    v18 = *(_QWORD *)v8;
    v19 = *(_QWORD *)v8 + 16LL * *(unsigned int *)(v8 + 8);
    while ( v19 != v18 )
    {
      v20 = *(_QWORD *)(v18 + 8);
      v21 = *(_DWORD *)v18;
      v18 += 16;
      sub_B99FD0(v12, v21, v20);
    }
  }
  v13 = *a1;
  v43 = 257;
  v14 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64))(**(_QWORD **)(v13 + 80) + 16LL))(
          *(_QWORD *)(v13 + 80),
          29,
          v9,
          v12);
  if ( !v14 )
  {
    v45 = 257;
    v14 = sub_B504D0(29, v9, v12, (__int64)v44, 0, 0);
    (*(void (__fastcall **)(_QWORD, __int64, _DWORD *, _QWORD, _QWORD))(**(_QWORD **)(v13 + 88) + 16LL))(
      *(_QWORD *)(v13 + 88),
      v14,
      v42,
      *(_QWORD *)(v13 + 56),
      *(_QWORD *)(v13 + 64));
    v22 = 16LL * *(unsigned int *)(v13 + 8);
    v23 = *(_QWORD *)v13;
    v24 = v23 + v22;
    while ( v24 != v23 )
    {
      v25 = *(_QWORD *)(v23 + 8);
      v26 = *(_DWORD *)v23;
      v23 += 16;
      sub_B99FD0(v14, v26, v25);
    }
  }
  v15 = *a1;
  v16 = a1[1];
  v42[1] = 0;
  v41 = v14;
  v45 = 257;
  return sub_B33D10(v15, a2, v16, 1, (int)&v41, 1, v42[0], (__int64)v44);
}
