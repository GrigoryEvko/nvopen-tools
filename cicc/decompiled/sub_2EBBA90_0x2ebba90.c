// Function: sub_2EBBA90
// Address: 0x2ebba90
//
void __fastcall sub_2EBBA90(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v8; // rcx
  unsigned int v9; // eax
  unsigned int v10; // ebx
  __int64 v11; // r14
  __int64 *v12; // r13
  __int64 v13; // rcx
  unsigned int v14; // eax
  __int64 v15; // rax
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // rdx
  unsigned int v19; // eax
  __int64 *v20; // rbx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // r9
  __int64 v28; // rbx
  __int64 v29; // rdx
  __int64 v30; // rcx
  __int64 v31; // r8
  __int64 v32; // r9
  __int64 *v33; // r14
  __int64 v34; // rdi
  unsigned int v35; // eax
  __int64 v36; // r8
  __int64 v37; // r9
  __int64 *v38; // rdx
  __int64 *v39; // [rsp+0h] [rbp-A0h]
  __int64 *v40; // [rsp+8h] [rbp-98h]
  unsigned int v41; // [rsp+14h] [rbp-8Ch]
  __int64 *v42; // [rsp+18h] [rbp-88h]
  __int64 *v43; // [rsp+20h] [rbp-80h] BYREF
  int v44; // [rsp+28h] [rbp-78h]
  _BYTE v45[112]; // [rsp+30h] [rbp-70h] BYREF

  if ( a3 )
  {
    v8 = (unsigned int)(*(_DWORD *)(a3 + 24) + 1);
    v9 = *(_DWORD *)(a3 + 24) + 1;
  }
  else
  {
    v8 = 0;
    v9 = 0;
  }
  v10 = *(_DWORD *)(a1 + 56);
  if ( v9 >= v10 )
    return;
  v11 = *(_QWORD *)(a1 + 48);
  v12 = *(__int64 **)(v11 + 8 * v8);
  if ( !v12 )
    return;
  if ( a4 )
  {
    v13 = (unsigned int)(*(_DWORD *)(a4 + 24) + 1);
    v14 = *(_DWORD *)(a4 + 24) + 1;
  }
  else
  {
    v13 = 0;
    v14 = 0;
  }
  if ( v10 <= v14 )
    return;
  v42 = *(__int64 **)(v11 + 8 * v13);
  if ( !v42 )
    return;
  v15 = sub_2EB3BB0(a1, a3, a4);
  if ( v15 )
  {
    v18 = (unsigned int)(*(_DWORD *)(v15 + 24) + 1);
    v19 = *(_DWORD *)(v15 + 24) + 1;
  }
  else
  {
    v18 = 0;
    v19 = 0;
  }
  if ( v10 <= v19 || v42 != *(__int64 **)(v11 + 8 * v18) )
  {
    *(_BYTE *)(a1 + 136) = 0;
    if ( v12 == (__int64 *)v42[1] )
    {
      v28 = *v42;
      sub_2EB5530(&v43, *v42, a2, v16, v17);
      v33 = v43;
      v39 = v43;
      v40 = &v43[v44];
      if ( v43 == v40 )
      {
LABEL_33:
        if ( v39 != (__int64 *)v45 )
          _libc_free((unsigned __int64)v39);
        sub_2E6D5A0(a1, *v42, v29, v30, v31, v32);
        v38 = 0;
        if ( *(_DWORD *)(a1 + 56) )
          v38 = **(__int64 ***)(a1 + 48);
        sub_2EBAF20(a1, a2, v38, (__int64)v42, v36, v37);
        goto LABEL_15;
      }
      v41 = *(_DWORD *)(a1 + 56);
      while ( 1 )
      {
        v29 = *v33;
        if ( *v33 )
        {
          v34 = (unsigned int)(*(_DWORD *)(v29 + 24) + 1);
          v35 = *(_DWORD *)(v29 + 24) + 1;
        }
        else
        {
          v34 = 0;
          v35 = 0;
        }
        if ( v35 < v41 && *(_QWORD *)(*(_QWORD *)(a1 + 48) + 8 * v34) && v28 != sub_2EB3BB0(a1, v28, v29) )
          break;
        if ( v40 == ++v33 )
          goto LABEL_33;
      }
      if ( v39 != (__int64 *)v45 )
        _libc_free((unsigned __int64)v39);
    }
    sub_2EBA950(a1, a2, *v12, *v42);
  }
LABEL_15:
  v20 = (__int64 *)(*(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8));
  if ( v20 != sub_2EB5800(*(__int64 **)a1, v20, a2, v16, v17) )
  {
    sub_2EB9A60(&v43, a1, a2, v21, v22, v23);
    if ( !(unsigned __int8)sub_2EB4750(a1, (__int64)&v43, v24, v25, v26, v27) )
      sub_2EBA1B0(a1, a2);
    if ( v43 != (__int64 *)v45 )
      _libc_free((unsigned __int64)v43);
  }
}
