// Function: sub_234A6B0
// Address: 0x234a6b0
//
void __fastcall sub_234A6B0(unsigned __int64 *a1)
{
  unsigned __int64 v1; // r12
  __int64 v2; // rbx
  unsigned __int64 v3; // r13
  __int64 v4; // r15
  unsigned __int64 v5; // r14
  unsigned __int64 v6; // rdi
  __int64 v7; // [rsp+0h] [rbp-60h]
  unsigned __int64 v8; // [rsp+8h] [rbp-58h]
  unsigned __int64 v10; // [rsp+18h] [rbp-48h]
  unsigned __int64 v11; // [rsp+20h] [rbp-40h]
  __int64 v12; // [rsp+28h] [rbp-38h]

  v8 = a1[1];
  v11 = *a1;
  if ( v8 != *a1 )
  {
    do
    {
      v7 = *(_QWORD *)(v11 + 24);
      v10 = *(_QWORD *)(v11 + 16);
      if ( v7 != v10 )
      {
        do
        {
          v1 = *(_QWORD *)(v10 + 16);
          v12 = *(_QWORD *)(v10 + 24);
          if ( v12 != v1 )
          {
            do
            {
              v2 = *(_QWORD *)(v1 + 24);
              v3 = *(_QWORD *)(v1 + 16);
              if ( v2 != v3 )
              {
                do
                {
                  v4 = *(_QWORD *)(v3 + 24);
                  v5 = *(_QWORD *)(v3 + 16);
                  if ( v4 != v5 )
                  {
                    do
                    {
                      v6 = v5 + 16;
                      v5 += 40LL;
                      sub_234A6B0(v6);
                    }
                    while ( v4 != v5 );
                    v5 = *(_QWORD *)(v3 + 16);
                  }
                  if ( v5 )
                    j_j___libc_free_0(v5);
                  v3 += 40LL;
                }
                while ( v2 != v3 );
                v3 = *(_QWORD *)(v1 + 16);
              }
              if ( v3 )
                j_j___libc_free_0(v3);
              v1 += 40LL;
            }
            while ( v12 != v1 );
            v1 = *(_QWORD *)(v10 + 16);
          }
          if ( v1 )
            j_j___libc_free_0(v1);
          v10 += 40LL;
        }
        while ( v7 != v10 );
        v10 = *(_QWORD *)(v11 + 16);
      }
      if ( v10 )
        j_j___libc_free_0(v10);
      v11 += 40LL;
    }
    while ( v8 != v11 );
    v11 = *a1;
  }
  if ( v11 )
    j_j___libc_free_0(v11);
}
