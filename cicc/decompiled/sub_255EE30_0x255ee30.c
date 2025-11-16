// Function: sub_255EE30
// Address: 0x255ee30
//
__int64 __fastcall sub_255EE30(__int64 a1, __int64 a2)
{
  __int64 v4; // rsi
  __int64 v5; // rbx
  __int64 v6; // r12
  __int64 result; // rax
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 v10; // rbx
  __int64 v11; // r14
  __int64 v12; // r12
  __int64 v13; // r13
  unsigned int v14; // eax
  __int64 v15; // [rsp+0h] [rbp-40h] BYREF
  int v16; // [rsp+8h] [rbp-38h]
  __int64 v17; // [rsp+10h] [rbp-30h] BYREF
  int v18; // [rsp+18h] [rbp-28h]

  v4 = *(unsigned int *)(a1 + 24);
  if ( (_DWORD)v4 )
  {
    v5 = *(_QWORD *)(a1 + 8);
    v6 = v5 + 16 * v4;
    do
    {
      if ( *(_DWORD *)(v5 + 8) > 0x40u && *(_QWORD *)v5 )
        j_j___libc_free_0_0(*(_QWORD *)v5);
      v5 += 16;
    }
    while ( v6 != v5 );
    v4 = *(unsigned int *)(a1 + 24);
  }
  result = sub_C7D6A0(*(_QWORD *)(a1 + 8), 16 * v4, 8);
  v8 = *(unsigned int *)(a2 + 24);
  *(_DWORD *)(a1 + 24) = v8;
  if ( (_DWORD)v8 )
  {
    v9 = sub_C7D670(16 * v8, 8);
    v16 = 0;
    *(_QWORD *)(a1 + 8) = v9;
    v10 = v9;
    LODWORD(v9) = *(_DWORD *)(a2 + 16);
    v15 = -1;
    *(_DWORD *)(a1 + 16) = v9;
    LODWORD(v9) = *(_DWORD *)(a2 + 20);
    v18 = 0;
    *(_DWORD *)(a1 + 20) = v9;
    v11 = *(unsigned int *)(a1 + 24);
    v12 = *(_QWORD *)(a2 + 8);
    v17 = -2;
    v13 = 0;
    if ( v11 )
    {
      do
      {
        v14 = *(_DWORD *)(v12 + 8);
        *(_DWORD *)(v10 + 8) = v14;
        if ( v14 <= 0x40 )
          *(_QWORD *)v10 = *(_QWORD *)v12;
        else
          sub_C43780(v10, (const void **)v12);
        ++v13;
        v10 += 16;
        v12 += 16;
      }
      while ( v11 != v13 );
    }
    sub_969240(&v17);
    return sub_969240(&v15);
  }
  else
  {
    *(_QWORD *)(a1 + 8) = 0;
    *(_QWORD *)(a1 + 16) = 0;
  }
  return result;
}
