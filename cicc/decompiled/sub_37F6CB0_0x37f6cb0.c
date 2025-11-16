// Function: sub_37F6CB0
// Address: 0x37f6cb0
//
__int64 __fastcall sub_37F6CB0(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 *v7; // rax
  unsigned __int64 *v8; // r12
  _QWORD *v9; // rdx
  unsigned __int64 v10; // rdi
  __int64 *v11; // rbx
  __int64 *v12; // r15
  __int64 v13; // rax
  unsigned __int64 *v14; // rax
  unsigned __int64 v15; // r14
  int v16; // ebx
  __int64 v18; // [rsp+8h] [rbp-58h]
  unsigned __int64 *v19; // [rsp+10h] [rbp-50h]
  unsigned __int64 *v20; // [rsp+18h] [rbp-48h]
  unsigned __int64 v21[7]; // [rsp+28h] [rbp-38h] BYREF

  v19 = (unsigned __int64 *)(a1 + 16);
  v18 = sub_C8D7D0(a1, a1 + 16, a2, 0x18u, v21, a6);
  v7 = *(unsigned __int64 **)a1;
  v8 = (unsigned __int64 *)(*(_QWORD *)a1 + 24LL * *(unsigned int *)(a1 + 8));
  if ( *(unsigned __int64 **)a1 != v8 )
  {
    v9 = (_QWORD *)v18;
    do
    {
      if ( v9 )
      {
        *v9 = *v7;
        v9[1] = v7[1];
        v9[2] = v7[2];
        v7[2] = 0;
        v7[1] = 0;
        *v7 = 0;
      }
      v7 += 3;
      v9 += 3;
    }
    while ( v8 != v7 );
    v20 = *(unsigned __int64 **)a1;
    v8 = (unsigned __int64 *)(*(_QWORD *)a1 + 24LL * *(unsigned int *)(a1 + 8));
    if ( *(unsigned __int64 **)a1 != v8 )
    {
      do
      {
        v10 = *(v8 - 3);
        v11 = (__int64 *)*(v8 - 2);
        v8 -= 3;
        v12 = (__int64 *)v10;
        if ( v11 != (__int64 *)v10 )
        {
          do
          {
            v13 = *v12;
            if ( *v12 )
            {
              if ( (v13 & 1) != 0 )
              {
                v14 = (unsigned __int64 *)(v13 & 0xFFFFFFFFFFFFFFFELL);
                v15 = (unsigned __int64)v14;
                if ( v14 )
                {
                  if ( (unsigned __int64 *)*v14 != v14 + 2 )
                    _libc_free(*v14);
                  j_j___libc_free_0(v15);
                }
              }
            }
            ++v12;
          }
          while ( v11 != v12 );
          v10 = *v8;
        }
        if ( v10 )
          j_j___libc_free_0(v10);
      }
      while ( v8 != v20 );
      v8 = *(unsigned __int64 **)a1;
    }
  }
  v16 = v21[0];
  if ( v19 != v8 )
    _libc_free((unsigned __int64)v8);
  *(_DWORD *)(a1 + 12) = v16;
  *(_QWORD *)a1 = v18;
  return v18;
}
