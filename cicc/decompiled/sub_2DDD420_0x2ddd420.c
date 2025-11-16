// Function: sub_2DDD420
// Address: 0x2ddd420
//
void __fastcall sub_2DDD420(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 *v6; // r14
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // r13
  __int64 v12; // rax
  unsigned __int64 *v13; // r12
  __int64 v14; // r15
  __int64 v15; // rdx
  __int64 v16; // rdi
  unsigned __int64 *v17; // r15
  int v18; // r15d
  __int64 v19; // [rsp+8h] [rbp-48h]
  unsigned __int64 v20[7]; // [rsp+18h] [rbp-38h] BYREF

  v6 = (unsigned __int64 *)(a1 + 16);
  v11 = sub_C8D7D0(a1, a1 + 16, a2, 0x30u, v20, a6);
  v12 = *(_QWORD *)a1;
  v13 = (unsigned __int64 *)(*(_QWORD *)a1 + 48LL * *(unsigned int *)(a1 + 8));
  if ( *(unsigned __int64 **)a1 != v13 )
  {
    v14 = v11;
    do
    {
      while ( 1 )
      {
        if ( v14 )
        {
          *(_DWORD *)(v14 + 8) = 0;
          *(_QWORD *)v14 = v14 + 16;
          *(_DWORD *)(v14 + 12) = 4;
          v15 = *(unsigned int *)(v12 + 8);
          if ( (_DWORD)v15 )
            break;
        }
        v12 += 48;
        v14 += 48;
        if ( v13 == (unsigned __int64 *)v12 )
          goto LABEL_7;
      }
      v16 = v14;
      v19 = v12;
      v14 += 48;
      sub_2DDB710(v16, v12, v15, v8, v9, v10);
      v12 = v19 + 48;
    }
    while ( v13 != (unsigned __int64 *)(v19 + 48) );
LABEL_7:
    v17 = *(unsigned __int64 **)a1;
    v13 = (unsigned __int64 *)(*(_QWORD *)a1 + 48LL * *(unsigned int *)(a1 + 8));
    if ( *(unsigned __int64 **)a1 != v13 )
    {
      do
      {
        v13 -= 6;
        if ( (unsigned __int64 *)*v13 != v13 + 2 )
          _libc_free(*v13);
      }
      while ( v13 != v17 );
      v13 = *(unsigned __int64 **)a1;
    }
  }
  v18 = v20[0];
  if ( v6 != v13 )
    _libc_free((unsigned __int64)v13);
  *(_QWORD *)a1 = v11;
  *(_DWORD *)(a1 + 12) = v18;
}
