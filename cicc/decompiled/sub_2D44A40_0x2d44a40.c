// Function: sub_2D44A40
// Address: 0x2d44a40
//
__int64 __fastcall sub_2D44A40(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r14
  __int64 *v5; // rbx
  unsigned int v6; // r13d
  unsigned __int8 *v7; // r15
  __int64 v8; // r13
  __int64 v10; // rdi
  __int64 v11; // r13
  __int64 v12; // r15
  __int64 v13; // rdi
  __int64 v14; // rax
  __int64 v15; // rdi
  __int64 v16; // r13
  __int64 v17; // r15
  __int64 v18; // rdi
  __int64 v19; // r13
  __int64 v20; // r10
  __int64 v21; // rdi
  __int64 v22; // rbx
  __int64 v23; // r12
  __int64 v24; // rdx
  unsigned int v25; // esi
  __int64 v26; // rbx
  __int64 v27; // r13
  __int64 v28; // rdx
  unsigned int v29; // esi
  __int64 v30; // rbx
  __int64 v31; // r13
  __int64 v32; // rdx
  unsigned int v33; // esi
  __int64 v34; // r13
  __int64 v35; // rdx
  unsigned int v36; // esi
  __int64 v37; // rbx
  __int64 v38; // r12
  __int64 v39; // rdx
  unsigned int v40; // esi
  unsigned __int8 *v41; // [rsp+8h] [rbp-98h]
  __int64 v42; // [rsp+8h] [rbp-98h]
  __int64 v43; // [rsp+8h] [rbp-98h]
  __int64 v44; // [rsp+8h] [rbp-98h]
  __int64 v45; // [rsp+8h] [rbp-98h]
  __int64 v46; // [rsp+8h] [rbp-98h]
  _BYTE v47[32]; // [rsp+10h] [rbp-90h] BYREF
  __int16 v48; // [rsp+30h] [rbp-70h]
  _BYTE v49[32]; // [rsp+40h] [rbp-60h] BYREF
  __int16 v50; // [rsp+60h] [rbp-40h]

  v3 = a3;
  v5 = *(__int64 **)(a1 + 24);
  v41 = **(unsigned __int8 ***)(a1 + 8);
  v6 = **(_DWORD **)a1;
  if ( v6 <= 2 )
  {
    if ( !v6 )
    {
      v10 = *(_QWORD *)(a2 + 80);
      v48 = 257;
      v11 = v5[7];
      v12 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v10 + 16LL))(v10, 28, a3, v11);
      if ( !v12 )
      {
        v50 = 257;
        v12 = sub_B504D0(28, v3, v11, (__int64)v49, 0, 0);
        (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
          *(_QWORD *)(a2 + 88),
          v12,
          v47,
          *(_QWORD *)(a2 + 56),
          *(_QWORD *)(a2 + 64));
        v26 = *(_QWORD *)a2;
        v27 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
        if ( *(_QWORD *)a2 != v27 )
        {
          do
          {
            v28 = *(_QWORD *)(v26 + 8);
            v29 = *(_DWORD *)v26;
            v26 += 16;
            sub_B99FD0(v12, v29, v28);
          }
          while ( v27 != v26 );
        }
      }
      v13 = *(_QWORD *)(a2 + 80);
      v48 = 257;
      v8 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, unsigned __int8 *))(*(_QWORD *)v13 + 16LL))(
             v13,
             29,
             v12,
             v41);
      if ( !v8 )
      {
        v50 = 257;
        v8 = sub_B504D0(29, v12, (__int64)v41, (__int64)v49, 0, 0);
        (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
          *(_QWORD *)(a2 + 88),
          v8,
          v47,
          *(_QWORD *)(a2 + 56),
          *(_QWORD *)(a2 + 64));
        v37 = *(_QWORD *)a2;
        v38 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
        while ( v38 != v37 )
        {
          v39 = *(_QWORD *)(v37 + 8);
          v40 = *(_DWORD *)v37;
          v37 += 16;
          sub_B99FD0(v8, v40, v39);
        }
      }
      return v8;
    }
    goto LABEL_12;
  }
  if ( v6 == 4 )
  {
LABEL_12:
    v14 = sub_2A2C8D0(v6, a2, a3, v41);
    v15 = *(_QWORD *)(a2 + 80);
    v48 = 257;
    v16 = v14;
    v42 = v5[6];
    v17 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v15 + 16LL))(v15, 28, v14, v42);
    if ( !v17 )
    {
      v50 = 257;
      v17 = sub_B504D0(28, v16, v42, (__int64)v49, 0, 0);
      (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
        *(_QWORD *)(a2 + 88),
        v17,
        v47,
        *(_QWORD *)(a2 + 56),
        *(_QWORD *)(a2 + 64));
      v46 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
      if ( *(_QWORD *)a2 != v46 )
      {
        v34 = *(_QWORD *)a2;
        do
        {
          v35 = *(_QWORD *)(v34 + 8);
          v36 = *(_DWORD *)v34;
          v34 += 16;
          sub_B99FD0(v17, v36, v35);
        }
        while ( v46 != v34 );
      }
    }
    v18 = *(_QWORD *)(a2 + 80);
    v48 = 257;
    v19 = v5[7];
    v20 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v18 + 16LL))(v18, 28, v3, v19);
    if ( !v20 )
    {
      v50 = 257;
      v44 = sub_B504D0(28, v3, v19, (__int64)v49, 0, 0);
      (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
        *(_QWORD *)(a2 + 88),
        v44,
        v47,
        *(_QWORD *)(a2 + 56),
        *(_QWORD *)(a2 + 64));
      v30 = *(_QWORD *)a2;
      v20 = v44;
      v31 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
      if ( *(_QWORD *)a2 != v31 )
      {
        do
        {
          v32 = *(_QWORD *)(v30 + 8);
          v33 = *(_DWORD *)v30;
          v30 += 16;
          v45 = v20;
          sub_B99FD0(v20, v33, v32);
          v20 = v45;
        }
        while ( v31 != v30 );
      }
    }
    v21 = *(_QWORD *)(a2 + 80);
    v48 = 257;
    v43 = v20;
    v8 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v21 + 16LL))(v21, 29, v20, v17);
    if ( !v8 )
    {
      v50 = 257;
      v8 = sub_B504D0(29, v43, v17, (__int64)v49, 0, 0);
      (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
        *(_QWORD *)(a2 + 88),
        v8,
        v47,
        *(_QWORD *)(a2 + 56),
        *(_QWORD *)(a2 + 64));
      v22 = *(_QWORD *)a2;
      v23 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
      while ( v23 != v22 )
      {
        v24 = *(_QWORD *)(v22 + 8);
        v25 = *(_DWORD *)v22;
        v22 += 16;
        sub_B99FD0(v8, v25, v24);
      }
    }
    return v8;
  }
  if ( v6 - 7 > 0xB )
    BUG();
  v7 = *(unsigned __int8 **)(**(_QWORD **)(a1 + 16) - 32LL);
  if ( *v5 != v5[1] )
    a3 = sub_2D44750((__int64 *)a2, a3, v5);
  v8 = sub_2A2C8D0(v6, a2, a3, v7);
  if ( *v5 != v5[1] )
    return sub_2D442D0((__int64 *)a2, v3, v8, v5);
  return v8;
}
