// Function: sub_3939810
// Address: 0x3939810
//
__int64 __fastcall sub_3939810(_QWORD *a1, __int64 *a2, __int64 a3, __int64 a4, int a5, __int64 a6)
{
  int v8; // r12d
  __int64 v9; // r8
  __int64 v10; // rbx
  __int64 v11; // rax
  unsigned __int64 v12; // rdx
  unsigned __int64 *v13; // r13
  unsigned __int64 v14; // r12
  _QWORD **v15; // rbx
  _QWORD **v16; // r15
  _QWORD *v17; // rdi
  _QWORD **v18; // r14
  _QWORD **v19; // rbx
  _QWORD **v20; // r15
  _QWORD *v21; // rdi
  _QWORD **v22; // r14
  __int64 v23; // rdx
  __int64 result; // rax
  __int64 v25; // rcx
  unsigned __int64 v27; // [rsp+8h] [rbp-48h]
  __int64 v29; // [rsp+18h] [rbp-38h]
  unsigned __int64 *v30; // [rsp+18h] [rbp-38h]

  v8 = a6;
  a1[6] = a6;
  a1[1] = 0;
  a1[2] = 0;
  a1[3] = 0;
  a1[4] = 0;
  a1[5] = 0;
  v9 = *a2;
  *a1 = &unk_4A3EF30;
  v10 = a2[1];
  v29 = v9;
  v11 = sub_22077B0(0x50u);
  if ( v11 )
  {
    *(_QWORD *)(v11 + 8) = v10;
    *(_QWORD *)(v11 + 16) = a2 + 2;
    *(_QWORD *)v11 = v29;
    *(_QWORD *)(v11 + 24) = a4;
    *(_QWORD *)(v11 + 32) = 0;
    *(_QWORD *)(v11 + 40) = 0;
    *(_QWORD *)(v11 + 48) = 0;
    *(_DWORD *)(v11 + 56) = a5;
    *(_DWORD *)(v11 + 60) = v8;
    *(_DWORD *)(v11 + 64) = 1;
    *(_QWORD *)(v11 + 72) = a3;
  }
  v12 = a1[1];
  a1[1] = v11;
  v27 = v12;
  if ( v12 )
  {
    v13 = *(unsigned __int64 **)(v12 + 32);
    v30 = *(unsigned __int64 **)(v12 + 40);
    if ( v30 != v13 )
    {
      do
      {
        v14 = v13[3];
        if ( v14 )
        {
          v15 = *(_QWORD ***)(v14 + 32);
          v16 = *(_QWORD ***)(v14 + 24);
          if ( v15 != v16 )
          {
            do
            {
              v17 = *v16;
              if ( v16 != *v16 )
              {
                while ( 1 )
                {
                  v18 = (_QWORD **)*v17;
                  j_j___libc_free_0((unsigned __int64)v17);
                  if ( v16 == v18 )
                    break;
                  v17 = v18;
                }
              }
              v16 += 3;
            }
            while ( v15 != v16 );
            v16 = *(_QWORD ***)(v14 + 24);
          }
          if ( v16 )
            j_j___libc_free_0((unsigned __int64)v16);
          v19 = *(_QWORD ***)(v14 + 8);
          v20 = *(_QWORD ***)v14;
          if ( v19 != *(_QWORD ***)v14 )
          {
            do
            {
              v21 = *v20;
              if ( v20 != *v20 )
              {
                while ( 1 )
                {
                  v22 = (_QWORD **)*v21;
                  j_j___libc_free_0((unsigned __int64)v21);
                  if ( v20 == v22 )
                    break;
                  v21 = v22;
                }
              }
              v20 += 3;
            }
            while ( v19 != v20 );
            v20 = *(_QWORD ***)v14;
          }
          if ( v20 )
            j_j___libc_free_0((unsigned __int64)v20);
          j_j___libc_free_0(v14);
        }
        if ( *v13 )
          j_j___libc_free_0(*v13);
        v13 += 7;
      }
      while ( v30 != v13 );
      v13 = *(unsigned __int64 **)(v27 + 32);
    }
    if ( v13 )
      j_j___libc_free_0((unsigned __int64)v13);
    j_j___libc_free_0(v27);
    v11 = a1[1];
  }
  v23 = *(_QWORD *)(v11 + 8);
  result = v11 + 32;
  v25 = *(_QWORD *)(result + 40);
  a1[3] = 0;
  a1[2] = v25;
  a1[4] = v23;
  a1[5] = result;
  return result;
}
