// Function: sub_1521F50
// Address: 0x1521f50
//
__int64 __fastcall sub_1521F50(__int64 *a1, unsigned int a2)
{
  __int64 v2; // r12
  unsigned __int64 v3; // rax
  __int64 v4; // r14
  __int64 v6; // rdx
  __int64 v7; // rcx
  unsigned __int64 v8; // r8
  __int64 v9; // rdi
  __int64 *v10; // r12
  unsigned __int64 v11; // rbx
  __int64 v12; // rdi
  __int64 v13[4]; // [rsp+0h] [rbp-B0h] BYREF
  __int64 v14[4]; // [rsp+20h] [rbp-90h] BYREF
  __int64 v15; // [rsp+40h] [rbp-70h] BYREF
  __int64 v16; // [rsp+48h] [rbp-68h]
  __int64 v17; // [rsp+50h] [rbp-60h]
  __int64 v18; // [rsp+58h] [rbp-58h]
  __int64 v19; // [rsp+60h] [rbp-50h]
  unsigned __int64 v20; // [rsp+68h] [rbp-48h]
  __int64 v21; // [rsp+70h] [rbp-40h]
  __int64 v22; // [rsp+78h] [rbp-38h]
  __int64 v23; // [rsp+80h] [rbp-30h]
  __int64 v24; // [rsp+88h] [rbp-28h]

  v2 = *a1;
  v3 = (__int64)(*(_QWORD *)(*a1 + 640) - *(_QWORD *)(*a1 + 632)) >> 4;
  if ( a2 < v3 )
    return sub_15197A0(*a1, a2);
  if ( a2 < *(_DWORD *)(v2 + 8) )
  {
    v4 = *(_QWORD *)(*(_QWORD *)v2 + 8LL * a2);
    if ( v4 )
      return v4;
  }
  if ( a2 < ((__int64)(*(_QWORD *)(v2 + 664) - *(_QWORD *)(v2 + 656)) >> 3) + v3 )
  {
    v15 = 0;
    v16 = 0;
    v17 = 0;
    v18 = 0;
    v19 = 0;
    v20 = 0;
    v21 = 0;
    v22 = 0;
    v23 = 0;
    v24 = 0;
    sub_1516C10(&v15, 0);
    sub_15201C0(v2, a2, (__int64)&v15);
    v4 = 0;
    sub_1520420(v2, (__int64)&v15, v6, v7, v8);
    if ( a2 < *(_DWORD *)(v2 + 8) )
      v4 = *(_QWORD *)(*(_QWORD *)v2 + 8LL * a2);
    v13[0] = v21;
    v13[1] = v22;
    v13[2] = v23;
    v13[3] = v24;
    v14[0] = v17;
    v14[1] = v18;
    v14[2] = v19;
    v14[3] = v20;
    sub_1514A90(v14, v13);
    v9 = v15;
    if ( v15 )
    {
      v10 = (__int64 *)v20;
      v11 = v24 + 8;
      if ( v24 + 8 > v20 )
      {
        do
        {
          v12 = *v10++;
          j_j___libc_free_0(v12, 512);
        }
        while ( v11 > (unsigned __int64)v10 );
        v9 = v15;
      }
      j_j___libc_free_0(v9, 8 * v16);
    }
    return v4;
  }
  return sub_1517EB0(*a1, a2);
}
