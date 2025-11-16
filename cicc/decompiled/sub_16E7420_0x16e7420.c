// Function: sub_16E7420
// Address: 0x16e7420
//
__int64 __fastcall sub_16E7420(__int64 a1, __int64 a2)
{
  _QWORD *v3; // rax
  _QWORD *v4; // rdx
  __int64 v5; // rbx
  __int64 v6; // rax
  __int64 **v7; // rax
  __int64 *v8; // r12
  __int64 v9; // rbx
  __int64 v10; // rdi
  unsigned __int64 *v11; // rbx
  unsigned __int64 *v12; // r14
  unsigned __int64 v13; // rdi
  unsigned __int64 *v14; // rbx
  unsigned __int64 v15; // r14
  unsigned __int64 v16; // rdi
  unsigned __int64 v17; // rdi
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // rax
  __int64 **v23; // r14
  __int64 v24; // r12
  __int64 v25; // rax
  __int64 *v26; // rbx
  unsigned __int64 *v27; // rbx
  unsigned __int64 *v28; // r14
  unsigned __int64 v29; // rdi
  unsigned __int64 *v30; // rbx
  unsigned __int64 v31; // rdi
  __int64 v32; // rax
  __int64 v33; // rdi
  __int64 v34[5]; // [rsp+8h] [rbp-28h] BYREF

  while ( 1 )
  {
    v3 = (_QWORD *)sub_16F82C0(*(_QWORD *)(a1 + 80));
    v4 = *(_QWORD **)(a1 + 216);
    if ( v4 && (v5 = *v4) != 0 )
    {
      if ( v3 && *v3 && v4 == v3 )
        return 0;
      v6 = *(_QWORD *)(v5 + 112);
      if ( !v6 )
        goto LABEL_25;
    }
    else
    {
      if ( !v3 || !*v3 )
        return 0;
      v5 = *v4;
      v6 = *(_QWORD *)(*v4 + 112LL);
      if ( !v6 )
      {
LABEL_25:
        v6 = sub_16FC3B0(v5);
        *(_QWORD *)(v5 + 112) = v6;
        if ( !v6 )
        {
          v21 = sub_2241E50(v5, a2, v18, v19, v20);
          *(_DWORD *)(a1 + 96) = 22;
          *(_QWORD *)(a1 + 104) = v21;
          return 0;
        }
      }
    }
    if ( *(_DWORD *)(v6 + 32) )
      break;
    if ( (unsigned __int8)sub_16FD950(**(_QWORD **)(a1 + 216)) )
    {
      v23 = *(__int64 ***)(a1 + 216);
      v24 = **v23;
      v25 = sub_22077B0(168);
      v26 = (__int64 *)v25;
      if ( v25 )
      {
        a2 = v24;
        sub_16FF2B0(v25, v24);
      }
      v8 = *v23;
      *v23 = v26;
      if ( v8 )
      {
        sub_16E3AF0(v8[17]);
        v27 = (unsigned __int64 *)v8[3];
        v28 = &v27[*((unsigned int *)v8 + 8)];
        while ( v28 != v27 )
        {
          v29 = *v27++;
          _libc_free(v29);
        }
        v30 = (unsigned __int64 *)v8[9];
        v15 = (unsigned __int64)&v30[2 * *((unsigned int *)v8 + 20)];
        if ( v30 != (unsigned __int64 *)v15 )
        {
          do
          {
            v31 = *v30;
            v30 += 2;
            _libc_free(v31);
          }
          while ( (unsigned __int64 *)v15 != v30 );
          goto LABEL_16;
        }
LABEL_17:
        if ( (__int64 *)v15 != v8 + 11 )
          _libc_free(v15);
        v17 = v8[3];
        if ( (__int64 *)v17 != v8 + 5 )
          _libc_free(v17);
        a2 = 168;
        j_j___libc_free_0(v8, 168);
      }
    }
    else
    {
      v7 = *(__int64 ***)(a1 + 216);
      v8 = *v7;
      *v7 = 0;
      if ( v8 )
      {
        v9 = v8[17];
        while ( v9 )
        {
          sub_16E3AF0(*(_QWORD *)(v9 + 24));
          v10 = v9;
          v9 = *(_QWORD *)(v9 + 16);
          j_j___libc_free_0(v10, 64);
        }
        v11 = (unsigned __int64 *)v8[3];
        v12 = &v11[*((unsigned int *)v8 + 8)];
        while ( v12 != v11 )
        {
          v13 = *v11++;
          _libc_free(v13);
        }
        v14 = (unsigned __int64 *)v8[9];
        v15 = (unsigned __int64)&v14[2 * *((unsigned int *)v8 + 20)];
        if ( v14 != (unsigned __int64 *)v15 )
        {
          do
          {
            v16 = *v14;
            v14 += 2;
            _libc_free(v16);
          }
          while ( v14 != (unsigned __int64 *)v15 );
LABEL_16:
          v15 = v8[9];
          goto LABEL_17;
        }
        goto LABEL_17;
      }
    }
  }
  sub_16E6D40(v34, a1, v6);
  v32 = v34[0];
  v33 = *(_QWORD *)(a1 + 88);
  v34[0] = 0;
  *(_QWORD *)(a1 + 88) = v32;
  if ( v33 )
  {
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v33 + 16LL))(v33);
    if ( v34[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v34[0] + 16LL))(v34[0]);
    v32 = *(_QWORD *)(a1 + 88);
  }
  *(_QWORD *)(a1 + 264) = v32;
  return 1;
}
