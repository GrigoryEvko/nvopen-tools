// Function: sub_2E73A60
// Address: 0x2e73a60
//
void __fastcall sub_2E73A60(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v7; // rsi
  unsigned int v8; // eax
  unsigned int v9; // edx
  __int64 v10; // rcx
  __int64 *v11; // r10
  __int64 v12; // rsi
  unsigned int v13; // eax
  __int64 *v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 *v17; // rbx
  __int64 *v18; // r13
  __int64 v19; // rax
  __int64 v20; // rdx
  unsigned int v21; // eax
  __int64 *v22; // rcx
  __int64 *v23; // [rsp+0h] [rbp-C0h] BYREF
  __int64 v24; // [rsp+8h] [rbp-B8h]
  _BYTE v25[176]; // [rsp+10h] [rbp-B0h] BYREF

  if ( a3 )
  {
    v7 = (unsigned int)(*(_DWORD *)(a3 + 24) + 1);
    v8 = *(_DWORD *)(a3 + 24) + 1;
  }
  else
  {
    v7 = 0;
    v8 = 0;
  }
  v9 = *(_DWORD *)(a1 + 32);
  if ( v8 < v9 )
  {
    v10 = *(_QWORD *)(a1 + 24);
    v11 = *(__int64 **)(v10 + 8 * v7);
    if ( v11 )
    {
      *(_BYTE *)(a1 + 112) = 0;
      if ( a4 )
      {
        v12 = (unsigned int)(*(_DWORD *)(a4 + 24) + 1);
        v13 = *(_DWORD *)(a4 + 24) + 1;
      }
      else
      {
        v12 = 0;
        v13 = 0;
      }
      if ( v9 > v13 && (v14 = *(__int64 **)(v10 + 8 * v12)) != 0 )
      {
        sub_2E73150(a1, a2, *v11, v14, a5, a4);
      }
      else
      {
        v23 = (__int64 *)v25;
        v24 = 0x800000000LL;
        sub_2E6FEF0(a1, a2, a4, v11, (__int64)&v23, a4);
        v17 = v23;
        v18 = &v23[2 * (unsigned int)v24];
        if ( v23 != v18 )
        {
          do
          {
            v19 = *v17;
            if ( *v17 )
            {
              v20 = (unsigned int)(*(_DWORD *)(v19 + 24) + 1);
              v21 = *(_DWORD *)(v19 + 24) + 1;
            }
            else
            {
              v20 = 0;
              v21 = 0;
            }
            if ( v21 >= *(_DWORD *)(a1 + 32) )
              BUG();
            v22 = (__int64 *)v17[1];
            v17 += 2;
            sub_2E73150(a1, a2, **(_QWORD **)(*(_QWORD *)(a1 + 24) + 8 * v20), v22, v15, v16);
          }
          while ( v18 != v17 );
          v18 = v23;
        }
        if ( v18 != (__int64 *)v25 )
          _libc_free((unsigned __int64)v18);
      }
    }
  }
}
