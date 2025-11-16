// Function: sub_35B4BD0
// Address: 0x35b4bd0
//
void __fastcall sub_35B4BD0(__int64 a1, int a2, unsigned int a3)
{
  _QWORD *v3; // rdi
  __int64 v4; // rax
  __int64 *v5; // rbx
  unsigned __int64 *v6; // r12
  unsigned __int64 v7; // r13
  unsigned __int64 v8; // rdi
  char *v9; // rax
  __int64 v10; // rdx
  char v11; // al
  __int64 v12; // rdx
  __int64 v13; // rcx
  unsigned int v14; // r13d
  __int16 *v15; // r12
  __int64 v16; // rbx
  unsigned __int64 **v17; // rax
  unsigned __int64 *v18; // r15
  unsigned __int64 v19; // rax
  unsigned __int64 v20; // r14
  unsigned __int64 v21; // rdi
  unsigned __int64 v22; // rdi
  int v23; // edx
  unsigned __int16 *v24; // [rsp+8h] [rbp-68h]
  char *v26; // [rsp+18h] [rbp-58h]
  unsigned __int64 v28; // [rsp+28h] [rbp-48h]
  __int64 v29; // [rsp+30h] [rbp-40h]
  __int64 v30; // [rsp+38h] [rbp-38h]

  v3 = *(_QWORD **)(a1 + 16);
  if ( a2 < 0 )
    v4 = *(_QWORD *)(v3[7] + 16LL * (a2 & 0x7FFFFFFF) + 8);
  else
    v4 = *(_QWORD *)(v3[38] + 8LL * (unsigned int)a2);
  if ( v4 )
  {
    do
    {
      if ( (*(_BYTE *)(v4 + 4) & 1) == 0
        && (*(_BYTE *)(v4 + 4) & 2) == 0
        && ((*(_BYTE *)(v4 + 3) & 0x10) == 0 || (*(_DWORD *)v4 & 0xFFF00) != 0) )
      {
        *(_BYTE *)(v4 + 4) |= 1u;
      }
      v4 = *(_QWORD *)(v4 + 32);
    }
    while ( v4 );
    v3 = *(_QWORD **)(a1 + 16);
  }
  if ( (*(_QWORD *)(v3[48] + 8LL * (a3 >> 6)) & (1LL << a3)) == 0 )
  {
    v9 = sub_E922F0(*(_QWORD **)(a1 + 8), a3);
    v24 = (unsigned __int16 *)&v9[2 * v10];
    if ( v9 == (char *)v24 )
    {
      v3 = *(_QWORD **)(a1 + 16);
    }
    else
    {
      v26 = v9;
      v3 = *(_QWORD **)(a1 + 16);
      do
      {
        v29 = *(_QWORD *)(v3[38] + 8LL * *(unsigned __int16 *)v26);
        if ( v29 )
        {
          do
          {
            v11 = *(_BYTE *)(v29 + 4);
            if ( (v11 & 1) == 0
              && (v11 & 2) == 0
              && ((*(_BYTE *)(v29 + 3) & 0x10) == 0 || (*(_DWORD *)v29 & 0xFFF00) != 0) )
            {
              *(_BYTE *)(v29 + 4) |= 1u;
              v12 = *(_QWORD *)(*(_QWORD *)(a1 + 32) + 16LL);
              v30 = *(_QWORD *)(a1 + 32);
              v13 = *(_QWORD *)(v12 + 8);
              v14 = *(_DWORD *)(v13 + 24LL * *(unsigned int *)(v29 + 8) + 16) & 0xFFF;
              v15 = (__int16 *)(*(_QWORD *)(v12 + 56)
                              + 2LL * (*(_DWORD *)(v13 + 24LL * *(unsigned int *)(v29 + 8) + 16) >> 12));
              do
              {
                if ( !v15 )
                  break;
                v16 = 8LL * v14;
                v17 = (unsigned __int64 **)(v16 + *(_QWORD *)(v30 + 424));
                v18 = *v17;
                if ( *v17 )
                {
                  v19 = v18[12];
                  v28 = v19;
                  if ( v19 )
                  {
                    v20 = *(_QWORD *)(v19 + 16);
                    while ( v20 )
                    {
                      sub_35B4780(*(_QWORD *)(v20 + 24));
                      v21 = v20;
                      v20 = *(_QWORD *)(v20 + 16);
                      j_j___libc_free_0(v21);
                    }
                    j_j___libc_free_0(v28);
                  }
                  v22 = v18[8];
                  if ( (unsigned __int64 *)v22 != v18 + 10 )
                    _libc_free(v22);
                  if ( (unsigned __int64 *)*v18 != v18 + 2 )
                    _libc_free(*v18);
                  j_j___libc_free_0((unsigned __int64)v18);
                  v17 = (unsigned __int64 **)(v16 + *(_QWORD *)(v30 + 424));
                }
                *v17 = 0;
                v23 = *v15++;
                v14 += v23;
              }
              while ( (_WORD)v23 );
            }
            v29 = *(_QWORD *)(v29 + 32);
          }
          while ( v29 );
          v3 = *(_QWORD **)(a1 + 16);
        }
        v26 += 2;
      }
      while ( v24 != (unsigned __int16 *)v26 );
    }
  }
  sub_2EBECB0(v3, a2, a3);
  v5 = (__int64 *)(*(_QWORD *)(*(_QWORD *)(a1 + 32) + 152LL) + 8LL * (a2 & 0x7FFFFFFF));
  v6 = (unsigned __int64 *)*v5;
  if ( *v5 )
  {
    sub_2E0AFD0(*v5);
    v7 = v6[12];
    if ( v7 )
    {
      sub_35B4780(*(_QWORD *)(v7 + 16));
      j_j___libc_free_0(v7);
    }
    v8 = v6[8];
    if ( (unsigned __int64 *)v8 != v6 + 10 )
      _libc_free(v8);
    if ( (unsigned __int64 *)*v6 != v6 + 2 )
      _libc_free(*v6);
    j_j___libc_free_0((unsigned __int64)v6);
  }
  *v5 = 0;
}
