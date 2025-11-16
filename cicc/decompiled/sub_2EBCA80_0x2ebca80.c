// Function: sub_2EBCA80
// Address: 0x2ebca80
//
__int64 __fastcall sub_2EBCA80(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v7; // r12
  __int64 v8; // r13
  __int64 v9; // r14
  int v10; // edx
  __int64 v11; // rax
  __int64 v12; // rcx
  __int64 v13; // rdx
  __int64 v14; // rbx
  __int64 v15; // r15
  unsigned __int64 v16; // rdi
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // rdx
  __int64 *v22; // r12
  __int64 v23; // r15
  __int64 v24; // rsi
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 **v28; // r13
  __int64 **v29; // r15
  __int64 *v30; // r12
  void *v31; // rax
  __int64 *v32; // r13
  __int64 v33; // rax
  __int64 v34; // rdi
  __int64 v35; // r12
  __int64 v36; // r12
  __int64 v37; // rdi
  unsigned int v38; // r12d
  __int64 *v39; // rdi
  __int64 v41; // [rsp+8h] [rbp-48h]
  _QWORD *v42; // [rsp+8h] [rbp-48h]
  _QWORD *v43; // [rsp+8h] [rbp-48h]
  _QWORD *v45; // [rsp+18h] [rbp-38h]

  v7 = *(_QWORD **)(a2 + 48);
  v45 = &v7[*(unsigned int *)(a2 + 56)];
  if ( v7 != v45 )
  {
    while ( 2 )
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
              v10 = *(_DWORD *)(a1 + 12);
              *(_DWORD *)(a1 + 8) = 0;
              v11 = 0;
              if ( !v10 )
              {
                sub_C8D5F0(a1, (const void *)(a1 + 16), 1u, 8u, a5, a6);
                v11 = 8LL * *(unsigned int *)(a1 + 8);
              }
              *(_QWORD *)(*(_QWORD *)a1 + v11) = 0;
              v12 = *(unsigned int *)(a1 + 536);
              v13 = *(_QWORD *)(a1 + 528);
              ++*(_DWORD *)(a1 + 8);
              if ( v13 != v13 + 56 * v12 )
              {
                v41 = a1;
                v14 = v13 + 56 * v12;
                v15 = v13;
                do
                {
                  v14 -= 56;
                  v16 = *(_QWORD *)(v14 + 24);
                  v13 = v14 + 40;
                  if ( v16 != v14 + 40 )
                    _libc_free(v16);
                }
                while ( v15 != v14 );
                a1 = v41;
              }
              *(_DWORD *)(a1 + 536) = 0;
              v17 = sub_2EB5B40(a1, 0, v13, v12, a5, a6);
              *(_QWORD *)(v17 + 8) = 0x100000001LL;
              *(_DWORD *)v17 = 1;
              sub_2E6D5A0(a1, 0, v18, 0x100000001LL, v19, v20);
              if ( *(_QWORD *)a2 != *(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8) )
              {
                v42 = v7;
                v21 = 1;
                v22 = *(__int64 **)a2;
                v23 = *(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8);
                do
                {
                  v24 = *v22++;
                  sub_2EBC640(a1, v24, v21, v9, 1, 0);
                }
                while ( (__int64 *)v23 != v22 );
                v7 = v42;
              }
              v25 = *(_QWORD *)(v8 + 24);
              v26 = *(unsigned int *)(v8 + 32);
              v27 = v25 + 8 * v26;
              if ( v25 != v27 )
                break;
            }
          }
        }
        if ( v45 == ++v7 )
          return 1;
      }
      v43 = v7;
      v28 = *(__int64 ***)(v8 + 24);
      v29 = (__int64 **)(v25 + 8 * v26);
      do
      {
        v30 = *v28;
        if ( *(_DWORD *)sub_2EB5B40(a1, **v28, v26, v27, a5, a6) )
        {
          v31 = sub_CB72A0();
          v32 = v30;
          v33 = sub_904010((__int64)v31, "Child ");
          v34 = *v30;
          v35 = v33;
          if ( *v32 )
            sub_2E39560(v34, v33);
          else
            sub_904010(v33, "nullptr");
          v36 = sub_904010(v35, " reachable after its parent ");
          sub_2E39560(v9, v36);
          v37 = v36;
          v38 = 0;
          sub_904010(v37, " is removed!\n");
          v39 = (__int64 *)sub_CB72A0();
          if ( v39[4] != v39[2] )
            sub_CB5AE0(v39);
          return v38;
        }
        ++v28;
      }
      while ( v29 != v28 );
      v7 = v43 + 1;
      if ( v45 != v43 + 1 )
        continue;
      break;
    }
  }
  return 1;
}
