// Function: sub_23B3D80
// Address: 0x23b3d80
//
void __fastcall sub_23B3D80(_QWORD *a1)
{
  unsigned __int64 v1; // r13
  unsigned __int64 v2; // rdi
  __int64 v3; // r14
  __int64 v4; // rbx
  __int64 v5; // r14
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // rdi
  __int64 v8; // r12
  __int64 v9; // r12
  __int64 v10; // r15
  _QWORD *v11; // r8
  unsigned __int64 v12; // rdi
  __int64 v13; // r10
  unsigned __int64 v14; // rdi
  unsigned __int64 *v15; // r12
  unsigned __int64 *v16; // r15
  unsigned __int64 *v17; // rbx
  unsigned __int64 *v18; // r12
  __int64 v20; // [rsp+8h] [rbp-58h]
  __int64 v21; // [rsp+10h] [rbp-50h]
  __int64 v22; // [rsp+10h] [rbp-50h]
  _QWORD *v23; // [rsp+18h] [rbp-48h]
  _QWORD *v24; // [rsp+18h] [rbp-48h]
  __int64 v25; // [rsp+20h] [rbp-40h]
  __int64 v26; // [rsp+28h] [rbp-38h]

  v1 = a1[1];
  v20 = a1[2];
  *a1 = &unk_4A15DE8;
  if ( v20 != v1 )
  {
    do
    {
      v2 = *(_QWORD *)(v1 + 24);
      if ( *(_DWORD *)(v1 + 36) )
      {
        v3 = *(unsigned int *)(v1 + 32);
        if ( (_DWORD)v3 )
        {
          v4 = 0;
          v26 = 8 * v3;
          do
          {
            v5 = *(_QWORD *)(v2 + v4);
            if ( v5 != -8 && v5 )
            {
              v6 = *(_QWORD *)(v5 + 56);
              v25 = *(_QWORD *)v5 + 89LL;
              if ( v6 != v5 + 72 )
                j_j___libc_free_0(v6);
              v7 = *(_QWORD *)(v5 + 32);
              if ( *(_DWORD *)(v5 + 44) )
              {
                v8 = *(unsigned int *)(v5 + 40);
                if ( (_DWORD)v8 )
                {
                  v9 = 8 * v8;
                  v10 = 0;
                  do
                  {
                    v11 = *(_QWORD **)(v7 + v10);
                    if ( v11 != (_QWORD *)-8LL && v11 )
                    {
                      v12 = v11[5];
                      v13 = *v11 + 81LL;
                      if ( (_QWORD *)v12 != v11 + 7 )
                      {
                        v21 = *v11 + 81LL;
                        v23 = v11;
                        j_j___libc_free_0(v12);
                        v13 = v21;
                        v11 = v23;
                      }
                      v14 = v11[1];
                      if ( (_QWORD *)v14 != v11 + 3 )
                      {
                        v22 = v13;
                        v24 = v11;
                        j_j___libc_free_0(v14);
                        v13 = v22;
                        v11 = v24;
                      }
                      sub_C7D6A0((__int64)v11, v13, 8);
                      v7 = *(_QWORD *)(v5 + 32);
                    }
                    v10 += 8;
                  }
                  while ( v9 != v10 );
                }
              }
              _libc_free(v7);
              v15 = *(unsigned __int64 **)(v5 + 16);
              v16 = *(unsigned __int64 **)(v5 + 8);
              if ( v15 != v16 )
              {
                do
                {
                  if ( (unsigned __int64 *)*v16 != v16 + 2 )
                    j_j___libc_free_0(*v16);
                  v16 += 4;
                }
                while ( v15 != v16 );
                v16 = *(unsigned __int64 **)(v5 + 8);
              }
              if ( v16 )
                j_j___libc_free_0((unsigned __int64)v16);
              sub_C7D6A0(v5, v25, 8);
              v2 = *(_QWORD *)(v1 + 24);
            }
            v4 += 8;
          }
          while ( v26 != v4 );
        }
      }
      _libc_free(v2);
      v17 = *(unsigned __int64 **)(v1 + 8);
      v18 = *(unsigned __int64 **)v1;
      if ( v17 != *(unsigned __int64 **)v1 )
      {
        do
        {
          if ( (unsigned __int64 *)*v18 != v18 + 2 )
            j_j___libc_free_0(*v18);
          v18 += 4;
        }
        while ( v17 != v18 );
        v18 = *(unsigned __int64 **)v1;
      }
      if ( v18 )
        j_j___libc_free_0((unsigned __int64)v18);
      v1 += 48LL;
    }
    while ( v20 != v1 );
    v1 = a1[1];
  }
  if ( v1 )
    j_j___libc_free_0(v1);
}
