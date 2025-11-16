// Function: sub_226B440
// Address: 0x226b440
//
void __fastcall sub_226B440(__int64 a1, unsigned __int64 a2)
{
  __int64 v3; // rax
  unsigned __int64 *v4; // rbx
  unsigned __int64 *v5; // r15
  unsigned __int64 v6; // rdx
  _QWORD *v7; // r14
  _QWORD *v8; // r12
  __int64 v9; // rax
  unsigned __int64 v10; // rdi
  _QWORD *v11; // rbx
  _QWORD *v12; // r12
  __int64 v13; // rax
  _QWORD v14[2]; // [rsp+10h] [rbp-70h] BYREF
  __int64 v15; // [rsp+20h] [rbp-60h]
  __int64 v16; // [rsp+30h] [rbp-50h]
  __int64 v17; // [rsp+38h] [rbp-48h]
  __int64 v18; // [rsp+40h] [rbp-40h]

  sub_FD6240(a1, a2);
  v3 = *(unsigned int *)(a1 + 48);
  if ( (_DWORD)v3 )
  {
    v11 = *(_QWORD **)(a1 + 32);
    v14[0] = 0;
    v14[1] = 0;
    v15 = -4096;
    v12 = &v11[4 * v3];
    v16 = 0;
    v17 = 0;
    v18 = -8192;
    do
    {
      v13 = v11[2];
      if ( v13 != 0 && v13 != -4096 && v13 != -8192 )
        sub_BD60C0(v11);
      v11 += 4;
    }
    while ( v12 != v11 );
    if ( v15 != -4096 && v15 != 0 )
      sub_BD60C0(v14);
    LODWORD(v3) = *(_DWORD *)(a1 + 48);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 32), 32LL * (unsigned int)v3, 8);
  v4 = *(unsigned __int64 **)(a1 + 16);
  while ( (unsigned __int64 *)(a1 + 8) != v4 )
  {
    v5 = v4;
    v4 = (unsigned __int64 *)v4[1];
    v6 = *v5 & 0xFFFFFFFFFFFFFFF8LL;
    *v4 = v6 | *v4 & 7;
    *(_QWORD *)(v6 + 8) = v4;
    v7 = (_QWORD *)v5[6];
    v8 = (_QWORD *)v5[5];
    *v5 &= 7u;
    v5[1] = 0;
    if ( v7 != v8 )
    {
      do
      {
        v9 = v8[2];
        if ( v9 != -4096 && v9 != 0 && v9 != -8192 )
          sub_BD60C0(v8);
        v8 += 3;
      }
      while ( v7 != v8 );
      v8 = (_QWORD *)v5[5];
    }
    if ( v8 )
      j_j___libc_free_0((unsigned __int64)v8);
    v10 = v5[3];
    if ( v5 + 5 != (unsigned __int64 *)v10 )
      _libc_free(v10);
    j_j___libc_free_0((unsigned __int64)v5);
  }
}
