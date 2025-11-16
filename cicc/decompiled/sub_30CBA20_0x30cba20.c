// Function: sub_30CBA20
// Address: 0x30cba20
//
void __fastcall sub_30CBA20(_QWORD *a1)
{
  __int64 v2; // rdi
  unsigned __int64 v3; // r14
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi
  __int64 v6; // r15
  __int64 v7; // r15
  __int64 v8; // rbx
  _QWORD *v9; // r12
  unsigned __int64 *v10; // r9
  __int64 v11; // r8
  unsigned __int64 v12; // rdi
  unsigned __int64 *v13; // [rsp+0h] [rbp-40h]
  __int64 v14; // [rsp+8h] [rbp-38h]
  __int64 v15; // [rsp+8h] [rbp-38h]

  *a1 = &unk_4A325A8;
  v2 = a1[9];
  if ( v2 )
  {
    sub_36FDA80(v2, LODWORD(qword_5041368[8]) == 2);
    v3 = a1[9];
    if ( v3 )
    {
      v4 = *(_QWORD *)(v3 + 24);
      if ( v4 )
        j_j___libc_free_0(v4);
      v5 = *(_QWORD *)v3;
      if ( *(_DWORD *)(v3 + 12) )
      {
        v6 = *(unsigned int *)(v3 + 8);
        if ( (_DWORD)v6 )
        {
          v7 = 8 * v6;
          v8 = 0;
          do
          {
            v9 = *(_QWORD **)(v5 + v8);
            if ( v9 != (_QWORD *)-8LL && v9 )
            {
              v10 = (unsigned __int64 *)v9[1];
              v11 = *v9 + 17LL;
              if ( v10 )
              {
                if ( (unsigned __int64 *)*v10 != v10 + 2 )
                {
                  v13 = (unsigned __int64 *)v9[1];
                  v14 = *v9 + 17LL;
                  _libc_free(*v10);
                  v10 = v13;
                  v11 = v14;
                }
                v15 = v11;
                j_j___libc_free_0((unsigned __int64)v10);
                v11 = v15;
              }
              sub_C7D6A0((__int64)v9, v11, 8);
              v5 = *(_QWORD *)v3;
            }
            v8 += 8;
          }
          while ( v7 != v8 );
        }
      }
      _libc_free(v5);
      j_j___libc_free_0(v3);
    }
  }
  v12 = a1[5];
  if ( (_QWORD *)v12 != a1 + 7 )
    j_j___libc_free_0(v12);
}
