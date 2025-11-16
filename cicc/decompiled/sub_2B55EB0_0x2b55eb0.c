// Function: sub_2B55EB0
// Address: 0x2b55eb0
//
__int64 __fastcall sub_2B55EB0(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // rdx
  __int64 v11; // r13
  unsigned __int64 v12; // rax
  __int64 v13; // rcx
  unsigned __int64 v14; // r12
  __int64 v15; // rdx
  __int64 v16; // rdx
  __int64 v17; // rdi
  unsigned __int64 v18; // r13
  __int64 v19; // rax
  unsigned __int64 v20; // r14
  unsigned __int64 *v21; // r15
  int v22; // r13d
  __int64 v24; // [rsp+8h] [rbp-58h]
  __int64 v25; // [rsp+10h] [rbp-50h]
  unsigned __int64 v26; // [rsp+18h] [rbp-48h]
  unsigned __int64 v27[7]; // [rsp+28h] [rbp-38h] BYREF

  v25 = a1 + 16;
  v7 = sub_C8D7D0(a1, a1 + 16, a2, 0x68u, v27, a6);
  v10 = *(unsigned int *)(a1 + 8);
  v24 = v7;
  v11 = v7;
  v12 = *(_QWORD *)a1;
  v13 = 3 * v10;
  v14 = *(_QWORD *)a1 + 104 * v10;
  if ( *(_QWORD *)a1 != v14 )
  {
    do
    {
      while ( 1 )
      {
        if ( v11 )
        {
          *(_QWORD *)v11 = *(_QWORD *)v12;
          *(_QWORD *)(v11 + 8) = *(_QWORD *)(v12 + 8);
          v15 = *(_QWORD *)(v12 + 16);
          *(_DWORD *)(v11 + 32) = 0;
          *(_QWORD *)(v11 + 16) = v15;
          *(_QWORD *)(v11 + 24) = v11 + 40;
          *(_DWORD *)(v11 + 36) = 1;
          v16 = *(unsigned int *)(v12 + 32);
          if ( (_DWORD)v16 )
            break;
        }
        v12 += 104LL;
        v11 += 104;
        if ( v14 == v12 )
          goto LABEL_7;
      }
      v17 = v11 + 24;
      v26 = v12;
      v11 += 104;
      sub_2B42980(v17, v12 + 24, v16, v13, v8, v9);
      v12 = v26 + 104;
    }
    while ( v14 != v26 + 104 );
LABEL_7:
    v18 = *(_QWORD *)a1;
    v14 = *(_QWORD *)a1 + 104LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v14 )
    {
      do
      {
        v19 = *(unsigned int *)(v14 - 72);
        v20 = *(_QWORD *)(v14 - 80);
        v14 -= 104LL;
        v19 <<= 6;
        v21 = (unsigned __int64 *)(v20 + v19);
        if ( v20 != v20 + v19 )
        {
          do
          {
            v21 -= 8;
            if ( (unsigned __int64 *)*v21 != v21 + 2 )
              _libc_free(*v21);
          }
          while ( (unsigned __int64 *)v20 != v21 );
          v20 = *(_QWORD *)(v14 + 24);
        }
        if ( v20 != v14 + 40 )
          _libc_free(v20);
      }
      while ( v14 != v18 );
      v14 = *(_QWORD *)a1;
    }
  }
  v22 = v27[0];
  if ( v25 != v14 )
    _libc_free(v14);
  *(_DWORD *)(a1 + 12) = v22;
  *(_QWORD *)a1 = v24;
  return v24;
}
