// Function: sub_2E8E920
// Address: 0x2e8e920
//
__int64 __fastcall sub_2E8E920(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  __int64 v8; // r12
  __int64 *v9; // rax
  __int64 v10; // rbx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // r13
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // r12
  __int64 v17; // rax
  unsigned __int64 v18; // rdx
  __int64 *v19; // r12
  unsigned int v20; // ebx
  __int64 v22; // r8
  __int64 v23; // r9
  unsigned __int64 v24; // rdx
  __int64 *v25; // [rsp+0h] [rbp-C0h] BYREF
  __int64 v26; // [rsp+8h] [rbp-B8h]
  _BYTE v27[176]; // [rsp+10h] [rbp-B0h] BYREF

  v26 = 0x1000000000LL;
  v6 = *a1;
  v25 = (__int64 *)v27;
  v7 = (*(_DWORD *)(v6 + 40) & 0xFFFFFFu) + 1;
  if ( (unsigned int)v7 > 0x10 )
  {
    sub_C8D5F0((__int64)&v25, v27, v7, 8u, a5, a6);
    v8 = *(unsigned __int16 *)(*a1 + 68);
    v24 = (unsigned int)v26 + 1LL;
    if ( v24 > HIDWORD(v26) )
      sub_C8D5F0((__int64)&v25, v27, v24, 8u, v22, v23);
    v9 = &v25[(unsigned int)v26];
  }
  else
  {
    v8 = *(unsigned __int16 *)(v6 + 68);
    v9 = (__int64 *)v27;
  }
  *v9 = v8;
  v10 = *(_QWORD *)(*a1 + 32);
  v11 = (unsigned int)(v26 + 1);
  v12 = *(_DWORD *)(*a1 + 40) & 0xFFFFFF;
  LODWORD(v26) = v26 + 1;
  v13 = v10 + 40 * v12;
  if ( v13 != v10 )
  {
    do
    {
      if ( *(_BYTE *)v10 || (*(_BYTE *)(v10 + 3) & 0x10) == 0 || *(int *)(v10 + 8) >= 0 )
      {
        v16 = sub_2EAE040(v10);
        v17 = (unsigned int)v26;
        v18 = (unsigned int)v26 + 1LL;
        if ( v18 > HIDWORD(v26) )
        {
          sub_C8D5F0((__int64)&v25, v27, v18, 8u, v14, v15);
          v17 = (unsigned int)v26;
        }
        v25[v17] = v16;
        LODWORD(v26) = v26 + 1;
      }
      v10 += 40;
    }
    while ( v13 != v10 );
    v11 = (unsigned int)v26;
  }
  v19 = v25;
  v20 = sub_C4ED70(v25, (__int64)&v25[v11]);
  if ( v19 != (__int64 *)v27 )
    _libc_free((unsigned __int64)v19);
  return v20;
}
