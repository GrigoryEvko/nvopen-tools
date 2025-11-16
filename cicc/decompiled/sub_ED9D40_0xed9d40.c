// Function: sub_ED9D40
// Address: 0xed9d40
//
__int64 __fastcall sub_ED9D40(_QWORD *a1, __int64 *a2, __int64 a3, __int64 a4, int a5, __int64 a6)
{
  int v8; // r12d
  __int64 v9; // r8
  __int64 v10; // rbx
  __int64 v11; // rax
  __int64 v12; // rdx
  _QWORD *v13; // r12
  _QWORD *v14; // r14
  _QWORD *v15; // r13
  _QWORD *v16; // rdi
  _QWORD *v17; // rbx
  _QWORD *v18; // r15
  __int64 v19; // rdi
  __int64 v20; // rdx
  __int64 result; // rax
  __int64 v22; // rcx
  __int64 v24; // [rsp+8h] [rbp-48h]
  __int64 v26; // [rsp+18h] [rbp-38h]
  _QWORD *v27; // [rsp+18h] [rbp-38h]

  v8 = a6;
  a1[6] = a6;
  a1[1] = 0;
  a1[2] = 0;
  a1[3] = 0;
  a1[4] = 0;
  a1[5] = 0;
  v9 = *a2;
  *a1 = &unk_49E4D18;
  v10 = a2[1];
  v26 = v9;
  v11 = sub_22077B0(80);
  if ( v11 )
  {
    *(_QWORD *)(v11 + 8) = v10;
    *(_QWORD *)(v11 + 16) = a2 + 2;
    *(_QWORD *)v11 = v26;
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
  v24 = v12;
  if ( v12 )
  {
    v13 = *(_QWORD **)(v12 + 32);
    v27 = *(_QWORD **)(v12 + 40);
    if ( v27 != v13 )
    {
      do
      {
        v14 = (_QWORD *)v13[6];
        if ( v14 )
        {
          v15 = v14 + 9;
          do
          {
            v16 = (_QWORD *)*(v15 - 3);
            v17 = (_QWORD *)*(v15 - 2);
            v15 -= 3;
            v18 = v16;
            if ( v17 != v16 )
            {
              do
              {
                if ( *v18 )
                  j_j___libc_free_0(*v18, v18[2] - *v18);
                v18 += 3;
              }
              while ( v17 != v18 );
              v16 = (_QWORD *)*v15;
            }
            if ( v16 )
              j_j___libc_free_0(v16, v15[2] - (_QWORD)v16);
          }
          while ( v14 != v15 );
          j_j___libc_free_0(v14, 72);
        }
        v19 = v13[3];
        if ( v19 )
          j_j___libc_free_0(v19, v13[5] - v19);
        if ( *v13 )
          j_j___libc_free_0(*v13, v13[2] - *v13);
        v13 += 10;
      }
      while ( v27 != v13 );
      v13 = *(_QWORD **)(v24 + 32);
    }
    if ( v13 )
      j_j___libc_free_0(v13, *(_QWORD *)(v24 + 48) - (_QWORD)v13);
    j_j___libc_free_0(v24, 80);
    v11 = a1[1];
  }
  v20 = *(_QWORD *)(v11 + 8);
  result = v11 + 32;
  v22 = *(_QWORD *)(result + 40);
  a1[3] = 0;
  a1[2] = v22;
  a1[4] = v20;
  a1[5] = result;
  return result;
}
