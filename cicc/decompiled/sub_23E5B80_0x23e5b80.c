// Function: sub_23E5B80
// Address: 0x23e5b80
//
void __fastcall sub_23E5B80(__int64 *a1, __int64 a2, _BYTE **a3)
{
  __int64 v4; // r12
  __int64 v5; // rdi
  __int64 v6; // r14
  __int64 v7; // r15
  unsigned int v8; // ebx
  __int64 *v9; // rdx
  __int64 v10; // rdi
  __int64 v11; // r14
  __int64 v12; // r10
  unsigned __int64 v13; // rdx
  __int64 v14; // rsi
  unsigned __int64 v15; // rax
  __int64 *v16; // rax
  __int64 v17; // rdi
  _BYTE *v18; // r9
  _QWORD *v19; // rdi
  __int64 v20; // r15
  __int64 v21; // rax
  _QWORD *v22; // rax
  unsigned int *v23; // rbx
  __int64 v24; // r14
  __int64 v25; // rdx
  unsigned int v26; // esi
  unsigned int *v27; // r15
  __int64 v28; // rdx
  unsigned int v29; // esi
  _QWORD *v30; // rax
  _QWORD *v31; // r10
  __int64 v32; // rdx
  unsigned int *v33; // rbx
  __int64 v34; // r14
  __int64 v35; // rdx
  unsigned int v36; // esi
  __int64 v37; // [rsp+0h] [rbp-A0h]
  __int64 v38; // [rsp+0h] [rbp-A0h]
  _BYTE *v39; // [rsp+8h] [rbp-98h]
  __int64 v40; // [rsp+8h] [rbp-98h]
  _BYTE *v41; // [rsp+8h] [rbp-98h]
  _QWORD *v42; // [rsp+8h] [rbp-98h]
  __int64 v43; // [rsp+8h] [rbp-98h]
  __int64 v44; // [rsp+8h] [rbp-98h]
  _BYTE *v45[4]; // [rsp+10h] [rbp-90h] BYREF
  __int16 v46; // [rsp+30h] [rbp-70h]
  _BYTE v47[32]; // [rsp+40h] [rbp-60h] BYREF
  __int16 v48; // [rsp+60h] [rbp-40h]

  v4 = *a1;
  v5 = *(_QWORD *)(a2 + 80);
  v46 = 257;
  v39 = *a3;
  v6 = **(_QWORD **)v4;
  v7 = (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v5 + 96LL))(v5, v6);
  if ( !v7 )
  {
    v48 = 257;
    v22 = sub_BD2C40(72, 2u);
    v7 = (__int64)v22;
    if ( v22 )
      sub_B4DE80((__int64)v22, v6, (__int64)v39, (__int64)v47, 0, 0);
    (*(void (__fastcall **)(_QWORD, __int64, _BYTE **, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
      *(_QWORD *)(a2 + 88),
      v7,
      v45,
      *(_QWORD *)(a2 + 56),
      *(_QWORD *)(a2 + 64));
    v23 = *(unsigned int **)a2;
    v24 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
    if ( *(_QWORD *)a2 != v24 )
    {
      do
      {
        v25 = *((_QWORD *)v23 + 1);
        v26 = *v23;
        v23 += 4;
        sub_B99FD0(v7, v26, v25);
      }
      while ( (unsigned int *)v24 != v23 );
    }
  }
  if ( *(_BYTE *)v7 == 17 )
  {
    v8 = *(_DWORD *)(v7 + 32);
    if ( v8 <= 0x40 )
    {
      if ( !*(_QWORD *)(v7 + 24) )
        return;
    }
    else if ( v8 == (unsigned int)sub_C444A0(v7 + 24) )
    {
      return;
    }
  }
  else
  {
    v14 = *(_QWORD *)(a2 + 56);
    if ( v14 )
      v14 -= 24;
    v15 = sub_F38250(v7, (__int64 *)(v14 + 24), 0, 0, 0, 0, 0, 0);
    sub_D5F1F0(a2, v15);
  }
  v9 = *(__int64 **)(v4 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(*v9 + 8) + 8LL) - 17 > 1 )
  {
    v16 = *(__int64 **)(v4 + 16);
    if ( *v16 )
    {
      v17 = *(_QWORD *)(a2 + 80);
      v46 = 257;
      v37 = *v16;
      v18 = (_BYTE *)(*(__int64 (__fastcall **)(__int64, __int64, _BYTE *, __int64, _QWORD, _QWORD))(*(_QWORD *)v17 + 32LL))(
                       v17,
                       17,
                       v39,
                       *v16,
                       0,
                       0);
      if ( !v18 )
      {
        v48 = 257;
        v40 = sub_B504D0(17, (__int64)v39, v37, (__int64)v47, 0, 0);
        (*(void (__fastcall **)(_QWORD, __int64, _BYTE **, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
          *(_QWORD *)(a2 + 88),
          v40,
          v45,
          *(_QWORD *)(a2 + 56),
          *(_QWORD *)(a2 + 64));
        v18 = (_BYTE *)v40;
        v38 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
        if ( *(_QWORD *)a2 != v38 )
        {
          v27 = *(unsigned int **)a2;
          do
          {
            v28 = *((_QWORD *)v27 + 1);
            v29 = *v27;
            v27 += 4;
            v41 = v18;
            sub_B99FD0((__int64)v18, v29, v28);
            v18 = v41;
          }
          while ( (unsigned int *)v38 != v27 );
        }
      }
      v19 = *(_QWORD **)(a2 + 72);
      v48 = 257;
      v20 = **(_QWORD **)(v4 + 8);
      v45[0] = v18;
      v21 = sub_BCB2B0(v19);
      v12 = sub_921130((unsigned int **)a2, v21, v20, v45, 1, (__int64)v47, 0);
    }
    else
    {
      v48 = 257;
      v45[0] = **(_BYTE ***)(v4 + 32);
      v45[1] = v39;
      v12 = sub_921130((unsigned int **)a2, **(_QWORD **)(v4 + 24), *v9, v45, 2, (__int64)v47, 0);
    }
  }
  else
  {
    v10 = *(_QWORD *)(a2 + 80);
    v46 = 257;
    v11 = *v9;
    v12 = (*(__int64 (__fastcall **)(__int64, __int64, _BYTE *))(*(_QWORD *)v10 + 96LL))(v10, *v9, v39);
    if ( !v12 )
    {
      v48 = 257;
      v30 = sub_BD2C40(72, 2u);
      v31 = v30;
      if ( v30 )
      {
        v32 = (__int64)v39;
        v42 = v30;
        sub_B4DE80((__int64)v30, v11, v32, (__int64)v47, 0, 0);
        v31 = v42;
      }
      v43 = (__int64)v31;
      (*(void (__fastcall **)(_QWORD, _QWORD *, _BYTE **, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
        *(_QWORD *)(a2 + 88),
        v31,
        v45,
        *(_QWORD *)(a2 + 56),
        *(_QWORD *)(a2 + 64));
      v33 = *(unsigned int **)a2;
      v12 = v43;
      v34 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
      if ( *(_QWORD *)a2 != v34 )
      {
        do
        {
          v35 = *((_QWORD *)v33 + 1);
          v36 = *v33;
          v33 += 4;
          v44 = v12;
          sub_B99FD0(v12, v36, v35);
          v12 = v44;
        }
        while ( (unsigned int *)v34 != v33 );
      }
    }
  }
  v13 = *(_QWORD *)(a2 + 56);
  if ( v13 )
    v13 -= 24LL;
  sub_23E5AA0(
    **(_QWORD **)(v4 + 40),
    **(_QWORD **)(v4 + 48),
    v13,
    v12,
    **(_WORD **)(v4 + 56),
    **(_DWORD **)(v4 + 64),
    **(_QWORD **)(v4 + 72),
    *(_QWORD *)(*(_QWORD *)(v4 + 72) + 8LL),
    **(_BYTE **)(v4 + 80),
    **(unsigned __int8 **)(v4 + 96),
    **(unsigned int **)(v4 + 104),
    *(_QWORD *)(v4 + 112));
}
