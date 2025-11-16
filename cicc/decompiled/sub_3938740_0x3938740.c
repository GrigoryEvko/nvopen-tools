// Function: sub_3938740
// Address: 0x3938740
//
void __fastcall sub_3938740(_QWORD *a1)
{
  unsigned __int64 v1; // rax
  unsigned __int64 *v2; // r13
  unsigned __int64 v3; // r12
  _QWORD **v4; // rbx
  _QWORD **v5; // r15
  _QWORD *v6; // rdi
  _QWORD **v7; // r14
  _QWORD **v8; // rbx
  _QWORD **v9; // r15
  _QWORD *v10; // rdi
  _QWORD **v11; // r14
  unsigned __int64 v13; // [rsp+10h] [rbp-40h]
  unsigned __int64 *v14; // [rsp+18h] [rbp-38h]

  *a1 = &unk_4A3EF30;
  v1 = a1[1];
  v13 = v1;
  if ( v1 )
  {
    v2 = *(unsigned __int64 **)(v1 + 32);
    v14 = *(unsigned __int64 **)(v1 + 40);
    if ( v14 != v2 )
    {
      do
      {
        v3 = v2[3];
        if ( v3 )
        {
          v4 = *(_QWORD ***)(v3 + 32);
          v5 = *(_QWORD ***)(v3 + 24);
          if ( v4 != v5 )
          {
            do
            {
              v6 = *v5;
              if ( *v5 != v5 )
              {
                while ( 1 )
                {
                  v7 = (_QWORD **)*v6;
                  j_j___libc_free_0((unsigned __int64)v6);
                  if ( v7 == v5 )
                    break;
                  v6 = v7;
                }
              }
              v5 += 3;
            }
            while ( v4 != v5 );
            v5 = *(_QWORD ***)(v3 + 24);
          }
          if ( v5 )
            j_j___libc_free_0((unsigned __int64)v5);
          v8 = *(_QWORD ***)(v3 + 8);
          v9 = *(_QWORD ***)v3;
          if ( v8 != *(_QWORD ***)v3 )
          {
            do
            {
              v10 = *v9;
              if ( *v9 != v9 )
              {
                while ( 1 )
                {
                  v11 = (_QWORD **)*v10;
                  j_j___libc_free_0((unsigned __int64)v10);
                  if ( v11 == v9 )
                    break;
                  v10 = v11;
                }
              }
              v9 += 3;
            }
            while ( v8 != v9 );
            v9 = *(_QWORD ***)v3;
          }
          if ( v9 )
            j_j___libc_free_0((unsigned __int64)v9);
          j_j___libc_free_0(v3);
        }
        if ( *v2 )
          j_j___libc_free_0(*v2);
        v2 += 7;
      }
      while ( v14 != v2 );
      v2 = *(unsigned __int64 **)(v13 + 32);
    }
    if ( v2 )
      j_j___libc_free_0((unsigned __int64)v2);
    j_j___libc_free_0(v13);
  }
  j_j___libc_free_0((unsigned __int64)a1);
}
