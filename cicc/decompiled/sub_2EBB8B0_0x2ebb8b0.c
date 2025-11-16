// Function: sub_2EBB8B0
// Address: 0x2ebb8b0
//
void __fastcall sub_2EBB8B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v8; // rcx
  unsigned int v9; // eax
  unsigned int v10; // edx
  __int64 *v11; // r9
  __int64 v12; // rcx
  unsigned int v13; // eax
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 *v17; // rbx
  __int64 *v18; // r13
  __int64 v19; // rsi
  unsigned int v20; // eax
  __int64 *v21; // rdx
  __int64 v22; // rax
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // r9
  __int64 v29; // [rsp+8h] [rbp-C8h]
  __int64 *v30; // [rsp+10h] [rbp-C0h] BYREF
  __int64 v31; // [rsp+18h] [rbp-B8h]
  _BYTE v32[176]; // [rsp+20h] [rbp-B0h] BYREF

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
  if ( v9 >= v10 || (v11 = *(__int64 **)(*(_QWORD *)(a1 + 48) + 8 * v8)) == 0 )
  {
    v24 = 0;
    if ( v10 )
      v24 = **(_QWORD **)(a1 + 48);
    v29 = sub_2EB4C20(a1, a3, v24);
    sub_2E6D5A0(a1, a3, v25, v26, v27, v28);
    v10 = *(_DWORD *)(a1 + 56);
    *(_BYTE *)(a1 + 136) = 0;
    v11 = (__int64 *)v29;
    if ( a4 )
      goto LABEL_6;
LABEL_21:
    v12 = 0;
    v13 = 0;
    goto LABEL_7;
  }
  *(_BYTE *)(a1 + 136) = 0;
  if ( !a4 )
    goto LABEL_21;
LABEL_6:
  v12 = (unsigned int)(*(_DWORD *)(a4 + 24) + 1);
  v13 = *(_DWORD *)(a4 + 24) + 1;
LABEL_7:
  if ( v13 < v10 && (v14 = *(_QWORD *)(*(_QWORD *)(a1 + 48) + 8 * v12)) != 0 )
  {
    sub_2EBAF20(a1, a2, v11, v14, a5, (__int64)v11);
  }
  else
  {
    v30 = (__int64 *)v32;
    v31 = 0x800000000LL;
    sub_2EB6550(a1, a2, a4, v11, (__int64)&v30, (__int64)v11);
    v17 = v30;
    v18 = &v30[2 * (unsigned int)v31];
    if ( v30 != v18 )
    {
      do
      {
        v22 = *v17;
        v23 = v17[1];
        if ( *v17 )
        {
          v19 = (unsigned int)(*(_DWORD *)(v22 + 24) + 1);
          v20 = *(_DWORD *)(v22 + 24) + 1;
        }
        else
        {
          v19 = 0;
          v20 = 0;
        }
        v21 = 0;
        if ( v20 < *(_DWORD *)(a1 + 56) )
          v21 = *(__int64 **)(*(_QWORD *)(a1 + 48) + 8 * v19);
        v17 += 2;
        sub_2EBAF20(a1, a2, v21, v23, v15, v16);
      }
      while ( v18 != v17 );
      v18 = v30;
    }
    if ( v18 != (__int64 *)v32 )
      _libc_free((unsigned __int64)v18);
  }
}
