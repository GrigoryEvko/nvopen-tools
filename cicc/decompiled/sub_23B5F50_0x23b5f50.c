// Function: sub_23B5F50
// Address: 0x23b5f50
//
void __fastcall sub_23B5F50(__int64 a1)
{
  int v2; // ecx
  unsigned __int64 v3; // rdi
  __int64 v4; // rax
  __int64 v5; // r15
  __int64 v6; // r12
  unsigned __int64 v7; // rdi
  unsigned __int64 v8; // rdi
  __int64 v9; // rax
  __int64 v10; // rbx
  __int64 v11; // r12
  unsigned __int64 v12; // rdi
  __int64 v13; // r14
  __int64 v14; // r13
  __int64 v15; // rbx
  __int64 v16; // r15
  _QWORD *v17; // r13
  unsigned __int64 v18; // rdi
  __int64 v19; // r14
  unsigned __int64 v20; // rdi
  unsigned __int64 v21; // rdi
  unsigned __int64 *v22; // rbx
  unsigned __int64 *v23; // r13
  __int64 v24; // [rsp+0h] [rbp-70h]
  __int64 v25; // [rsp+8h] [rbp-68h]
  __int64 v26; // [rsp+10h] [rbp-60h]
  __int64 v28; // [rsp+20h] [rbp-50h]
  __int64 v29; // [rsp+28h] [rbp-48h]
  __int64 v30; // [rsp+30h] [rbp-40h]
  __int64 v31; // [rsp+38h] [rbp-38h]

  v2 = *(_DWORD *)(a1 + 12);
  v3 = *(_QWORD *)a1;
  if ( v2 )
  {
    v4 = *(unsigned int *)(a1 + 8);
    if ( (_DWORD)v4 )
    {
      v5 = 0;
      v30 = 8 * v4;
      do
      {
        v6 = *(_QWORD *)(v3 + v5);
        if ( v6 != -8 && v6 )
        {
          v7 = *(_QWORD *)(v6 + 56);
          v28 = *(_QWORD *)v6 + 89LL;
          if ( v7 != v6 + 72 )
            j_j___libc_free_0(v7);
          v8 = *(_QWORD *)(v6 + 32);
          if ( *(_DWORD *)(v6 + 44) )
          {
            v9 = *(unsigned int *)(v6 + 40);
            if ( (_DWORD)v9 )
            {
              v29 = v6;
              v10 = 0;
              v31 = 8 * v9;
              v24 = v5;
              do
              {
                v11 = *(_QWORD *)(v8 + v10);
                if ( v11 != -8 && v11 )
                {
                  v12 = *(_QWORD *)(v11 + 72);
                  v13 = *(_QWORD *)v11 + 97LL;
                  if ( *(_DWORD *)(v11 + 84) )
                  {
                    v14 = *(unsigned int *)(v11 + 80);
                    if ( (_DWORD)v14 )
                    {
                      v26 = *(_QWORD *)v11 + 97LL;
                      v25 = v10;
                      v15 = 0;
                      v16 = 8 * v14;
                      do
                      {
                        v17 = *(_QWORD **)(v12 + v15);
                        if ( v17 && v17 != (_QWORD *)-8LL )
                        {
                          v18 = v17[1];
                          v19 = *v17 + 41LL;
                          if ( (_QWORD *)v18 != v17 + 3 )
                            j_j___libc_free_0(v18);
                          sub_C7D6A0((__int64)v17, v19, 8);
                          v12 = *(_QWORD *)(v11 + 72);
                        }
                        v15 += 8;
                      }
                      while ( v16 != v15 );
                      v13 = v26;
                      v10 = v25;
                    }
                  }
                  _libc_free(v12);
                  v20 = *(_QWORD *)(v11 + 40);
                  if ( v20 != v11 + 56 )
                    j_j___libc_free_0(v20);
                  v21 = *(_QWORD *)(v11 + 8);
                  if ( v21 != v11 + 24 )
                    j_j___libc_free_0(v21);
                  sub_C7D6A0(v11, v13, 8);
                  v8 = *(_QWORD *)(v29 + 32);
                }
                v10 += 8;
              }
              while ( v31 != v10 );
              v6 = v29;
              v5 = v24;
            }
          }
          _libc_free(v8);
          v22 = *(unsigned __int64 **)(v6 + 16);
          v23 = *(unsigned __int64 **)(v6 + 8);
          if ( v22 != v23 )
          {
            do
            {
              if ( (unsigned __int64 *)*v23 != v23 + 2 )
                j_j___libc_free_0(*v23);
              v23 += 4;
            }
            while ( v22 != v23 );
            v23 = *(unsigned __int64 **)(v6 + 8);
          }
          if ( v23 )
            j_j___libc_free_0((unsigned __int64)v23);
          sub_C7D6A0(v6, v28, 8);
          v3 = *(_QWORD *)a1;
        }
        v5 += 8;
      }
      while ( v30 != v5 );
    }
  }
  _libc_free(v3);
}
