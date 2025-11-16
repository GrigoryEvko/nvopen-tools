// Function: sub_2740590
// Address: 0x2740590
//
void __fastcall sub_2740590(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // r13
  unsigned __int64 *v10; // rax
  unsigned __int64 *v11; // r12
  __int64 v12; // r15
  __int64 v13; // rdx
  unsigned __int64 *v14; // r15
  int v15; // r15d
  unsigned __int64 *v16; // [rsp+8h] [rbp-48h]
  unsigned __int64 v17[7]; // [rsp+18h] [rbp-38h] BYREF

  v9 = sub_C8D7D0(a1, a1 + 16, a2, 0x90u, v17, a6);
  v10 = *(unsigned __int64 **)a1;
  v11 = (unsigned __int64 *)(*(_QWORD *)a1 + 144LL * *(unsigned int *)(a1 + 8));
  if ( *(unsigned __int64 **)a1 != v11 )
  {
    v12 = v9;
    do
    {
      if ( v12 )
      {
        *(_DWORD *)(v12 + 8) = 0;
        *(_QWORD *)v12 = v12 + 16;
        *(_DWORD *)(v12 + 12) = 8;
        v13 = *((unsigned int *)v10 + 2);
        if ( (_DWORD)v13 )
        {
          v16 = v10;
          sub_27389D0(v12, (char **)v10, v13, v6, v7, v8);
          v10 = v16;
        }
      }
      v10 += 18;
      v12 += 144;
    }
    while ( v11 != v10 );
    v14 = *(unsigned __int64 **)a1;
    v11 = (unsigned __int64 *)(*(_QWORD *)a1 + 144LL * *(unsigned int *)(a1 + 8));
    if ( *(unsigned __int64 **)a1 != v11 )
    {
      do
      {
        v11 -= 18;
        if ( (unsigned __int64 *)*v11 != v11 + 2 )
          _libc_free(*v11);
      }
      while ( v11 != v14 );
      v11 = *(unsigned __int64 **)a1;
    }
  }
  v15 = v17[0];
  if ( (unsigned __int64 *)(a1 + 16) != v11 )
    _libc_free((unsigned __int64)v11);
  *(_QWORD *)a1 = v9;
  *(_DWORD *)(a1 + 12) = v15;
}
