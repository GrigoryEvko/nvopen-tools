// Function: sub_D9AD80
// Address: 0xd9ad80
//
char __fastcall sub_D9AD80(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *v6; // rbx
  __int64 v7; // rax
  __int64 *i; // r14
  __int64 v9; // rsi
  char *v10; // rdx
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // rcx
  _QWORD *v14; // r14
  __int64 v15; // rbx
  __int64 v16; // rax
  unsigned __int64 v17; // rdx
  char **v18; // rsi
  __int64 v19; // r8
  __int64 v20; // r9
  char *v21; // rdi
  _QWORD *v23; // [rsp+18h] [rbp-78h]
  char *v24; // [rsp+20h] [rbp-70h] BYREF
  __int64 v25; // [rsp+28h] [rbp-68h]
  _BYTE v26[96]; // [rsp+30h] [rbp-60h] BYREF

  if ( *(_DWORD *)(a2 + 32) )
  {
    LOBYTE(v7) = sub_D92140(a1, a2, a3);
    if ( !(_BYTE)v7 )
    {
      v12 = *(_QWORD *)(a1 + 40);
      v13 = (__int64)&v24;
      v24 = v26;
      v14 = (_QWORD *)v12;
      v25 = 0x600000000LL;
      v23 = (_QWORD *)(v12 + 8LL * *(unsigned int *)(a1 + 48));
      if ( v23 != (_QWORD *)v12 )
      {
        do
        {
          v15 = *v14;
          if ( !(*(unsigned __int8 (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)a2 + 16LL))(a2, *v14, a3) )
          {
            v16 = (unsigned int)v25;
            v13 = HIDWORD(v25);
            v17 = (unsigned int)v25 + 1LL;
            if ( v17 > HIDWORD(v25) )
            {
              sub_C8D5F0((__int64)&v24, v26, v17, 8u, v11, v12);
              v16 = (unsigned int)v25;
            }
            v10 = v24;
            *(_QWORD *)&v24[8 * v16] = v15;
            LODWORD(v25) = v25 + 1;
          }
          ++v14;
        }
        while ( v23 != v14 );
      }
      v18 = &v24;
      sub_D91460(a1 + 40, &v24, (__int64)v10, v13, v11, v12);
      v7 = *(unsigned int *)(a1 + 48);
      if ( v7 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 52) )
      {
        v18 = (char **)(a1 + 56);
        sub_C8D5F0(a1 + 40, (const void *)(a1 + 56), v7 + 1, 8u, v19, v20);
        v7 = *(unsigned int *)(a1 + 48);
      }
      *(_QWORD *)(*(_QWORD *)(a1 + 40) + 8 * v7) = a2;
      v21 = v24;
      ++*(_DWORD *)(a1 + 48);
      if ( v21 != v26 )
        LOBYTE(v7) = _libc_free(v21, v18);
    }
  }
  else
  {
    v6 = *(__int64 **)(a2 + 40);
    v7 = *(unsigned int *)(a2 + 48);
    for ( i = &v6[v7]; i != v6; LOBYTE(v7) = sub_D9AD80(a1, v9, a3) )
      v9 = *v6++;
  }
  return v7;
}
