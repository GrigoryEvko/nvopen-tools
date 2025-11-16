// Function: sub_2E759D0
// Address: 0x2e759d0
//
__int64 __fastcall sub_2E759D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v7; // rbx
  __int64 v8; // r15
  __int64 v9; // r14
  __int64 v10; // rax
  __int64 v11; // rcx
  __int64 v12; // rdx
  __int64 v13; // r13
  unsigned __int64 v14; // rdi
  __int64 v15; // rcx
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 **v18; // r15
  __int64 *v19; // r13
  void *v20; // rax
  __int64 v21; // rax
  __int64 v22; // r12
  __int64 v23; // r12
  __int64 v24; // rdi
  unsigned int v25; // r12d
  __int64 *v26; // rdi
  __int64 *v28; // [rsp+10h] [rbp-40h]
  __int64 v29; // [rsp+18h] [rbp-38h]
  __int64 **v30; // [rsp+18h] [rbp-38h]

  v7 = *(__int64 **)(a2 + 24);
  v28 = &v7[*(unsigned int *)(a2 + 32)];
  if ( v7 == v28 )
  {
    return 1;
  }
  else
  {
    while ( 1 )
    {
      v8 = *v7;
      if ( *v7 )
      {
        v9 = *(_QWORD *)v8;
        if ( *(_QWORD *)v8 )
        {
          if ( *(_DWORD *)(v8 + 32) )
          {
            *(_DWORD *)(a1 + 8) = 0;
            v10 = 0;
            if ( !*(_DWORD *)(a1 + 12) )
            {
              sub_C8D5F0(a1, (const void *)(a1 + 16), 1u, 8u, a5, a6);
              v10 = 8LL * *(unsigned int *)(a1 + 8);
            }
            *(_QWORD *)(*(_QWORD *)a1 + v10) = 0;
            v11 = *(unsigned int *)(a1 + 536);
            ++*(_DWORD *)(a1 + 8);
            v12 = *(_QWORD *)(a1 + 528);
            v13 = v12 + 56 * v11;
            while ( v12 != v13 )
            {
              while ( 1 )
              {
                v13 -= 56;
                v14 = *(_QWORD *)(v13 + 24);
                if ( v14 == v13 + 40 )
                  break;
                v29 = v12;
                _libc_free(v14);
                v12 = v29;
                if ( v29 == v13 )
                  goto LABEL_13;
              }
            }
LABEL_13:
            *(_DWORD *)(a1 + 536) = 0;
            sub_2E755A0(a1, **(_QWORD **)a2, 0, v9, 0, 0);
            v16 = *(_QWORD *)(v8 + 24);
            v17 = v16 + 8LL * *(unsigned int *)(v8 + 32);
            v18 = (__int64 **)v16;
            v30 = (__int64 **)v17;
            if ( v16 != v17 )
              break;
          }
        }
      }
LABEL_3:
      if ( v28 == ++v7 )
        return 1;
    }
    while ( 1 )
    {
      v19 = *v18;
      if ( *(_DWORD *)sub_2E6F1C0(a1, **v18, v16, v15, a5, a6) )
        break;
      if ( v30 == ++v18 )
        goto LABEL_3;
    }
    v20 = sub_CB72A0();
    v21 = sub_904010((__int64)v20, "Child ");
    v22 = v21;
    if ( *v19 )
      sub_2E39560(*v19, v21);
    else
      sub_904010(v21, "nullptr");
    v23 = sub_904010(v22, " reachable after its parent ");
    sub_2E39560(v9, v23);
    v24 = v23;
    v25 = 0;
    sub_904010(v24, " is removed!\n");
    v26 = (__int64 *)sub_CB72A0();
    if ( v26[4] != v26[2] )
      sub_CB5AE0(v26);
  }
  return v25;
}
