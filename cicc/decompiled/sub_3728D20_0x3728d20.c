// Function: sub_3728D20
// Address: 0x3728d20
//
void __fastcall sub_3728D20(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r14
  __int64 v5; // r9
  unsigned __int64 v6; // r12
  int v7; // r8d
  _QWORD *v8; // rax
  _BYTE *v9; // rdx
  __int64 v10; // rcx
  _QWORD *i; // rdx
  int v12; // eax
  unsigned __int8 **v13; // rbx
  unsigned __int8 **v14; // r12
  unsigned __int8 *v15; // rsi
  __int64 v16; // rax
  __int64 *v17; // r12
  __int64 *v18; // rbx
  __int64 v19; // rax
  unsigned __int64 j; // rax
  __int64 v21; // rcx
  _BYTE *v22; // [rsp-248h] [rbp-248h] BYREF
  __int64 v23; // [rsp-240h] [rbp-240h]
  _BYTE v24[568]; // [rsp-238h] [rbp-238h] BYREF

  if ( *(_DWORD *)(a1 + 16) )
  {
    v4 = 0;
    (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(a2 + 224) + 176LL))(*(_QWORD *)(a2 + 224), a3, 0);
    if ( (unsigned __int16)sub_31DF670(a2) > 4u )
      v4 = sub_3728B90(a1, a2);
    (*(void (__fastcall **)(_QWORD, _QWORD, _QWORD))(**(_QWORD **)(a2 + 224) + 208LL))(
      *(_QWORD *)(a2 + 224),
      *(_QWORD *)(a1 + 40),
      0);
    v6 = *(unsigned int *)(a1 + 16);
    v22 = v24;
    v23 = 0x4000000000LL;
    v7 = v6;
    if ( v6 )
    {
      v8 = v24;
      v9 = v24;
      if ( v6 > 0x40 )
      {
        sub_C8D5F0((__int64)&v22, v24, v6, 8u, v6, v5);
        v9 = v22;
        v7 = v6;
        v8 = &v22[8 * (unsigned int)v23];
      }
      v10 = 8 * v6;
      for ( i = &v9[8 * v6]; i != v8; ++v8 )
      {
        if ( v8 )
          *v8 = 0;
      }
      v12 = *(_DWORD *)(a1 + 16);
      LODWORD(v23) = v7;
      if ( !v12 )
        goto LABEL_12;
      v16 = *(_QWORD *)(a1 + 8);
      v17 = (__int64 *)(v16 + 16LL * *(unsigned int *)(a1 + 24));
      if ( (__int64 *)v16 == v17 )
        goto LABEL_12;
      while ( 1 )
      {
        v18 = (__int64 *)v16;
        if ( *(_QWORD *)v16 != -4096 && *(_QWORD *)v16 != -8192 )
          break;
        v16 += 16;
        if ( v17 == (__int64 *)v16 )
          goto LABEL_12;
      }
      if ( (__int64 *)v16 == v17 )
      {
LABEL_12:
        v13 = (unsigned __int8 **)v22;
        v14 = (unsigned __int8 **)&v22[v10];
      }
      else
      {
        if ( !*(_BYTE *)(v16 + 12) )
          goto LABEL_36;
LABEL_28:
        v19 = sub_31DA6B0(a2);
        for ( j = (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v19 + 168LL))(v19, *v18);
              ;
              j = sub_E808D0(*v18, 0, *(_QWORD **)(a2 + 216), 0) )
        {
          v21 = *((unsigned int *)v18 + 2);
          v18 += 2;
          *(_QWORD *)&v22[8 * v21] = j;
          if ( v18 == v17 )
            break;
          while ( *v18 == -8192 || *v18 == -4096 )
          {
            v18 += 2;
            if ( v17 == v18 )
              goto LABEL_33;
          }
          if ( v18 == v17 )
            break;
          if ( *((_BYTE *)v18 + 12) )
            goto LABEL_28;
LABEL_36:
          ;
        }
LABEL_33:
        v13 = (unsigned __int8 **)v22;
        v14 = (unsigned __int8 **)&v22[8 * (unsigned int)v23];
      }
    }
    else
    {
      v14 = (unsigned __int8 **)v24;
      v13 = (unsigned __int8 **)v24;
    }
    while ( v14 != v13 )
    {
      v15 = *v13++;
      sub_E9A5B0(*(_QWORD *)(a2 + 224), v15);
    }
    if ( v4 )
      (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(a2 + 224) + 208LL))(*(_QWORD *)(a2 + 224), v4, 0);
    if ( v22 != v24 )
      _libc_free((unsigned __int64)v22);
  }
}
