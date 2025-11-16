// Function: sub_31166B0
// Address: 0x31166b0
//
void __fastcall sub_31166B0(unsigned __int64 **a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, unsigned __int64 a6)
{
  __int64 i; // r13
  int *v7; // r14
  int *v8; // r15
  int v9; // eax
  unsigned __int64 v10; // rbx
  unsigned __int64 v11; // r12
  unsigned __int64 v12; // rdi
  __int64 v13; // [rsp+18h] [rbp-68h] BYREF
  __int64 v14; // [rsp+20h] [rbp-60h] BYREF
  int v15; // [rsp+28h] [rbp-58h] BYREF
  unsigned __int64 v16; // [rsp+30h] [rbp-50h]
  int *v17; // [rsp+38h] [rbp-48h]
  int *v18; // [rsp+40h] [rbp-40h]
  __int64 v19; // [rsp+48h] [rbp-38h]

  v15 = 0;
  v16 = 0;
  v17 = &v15;
  v18 = &v15;
  v19 = 0;
  sub_3115D50(a1, &v14, a3, a4, a5, a6);
  LODWORD(v13) = v19;
  sub_CB6200(a2, (unsigned __int8 *)&v13, 4u);
  for ( i = (__int64)v17; (int *)i != &v15; i = sub_220EEE0(i) )
  {
    LODWORD(v13) = *(_DWORD *)(i + 32);
    sub_CB6200(a2, (unsigned __int8 *)&v13, 4u);
    v13 = *(_QWORD *)(i + 40);
    sub_CB6200(a2, (unsigned __int8 *)&v13, 8u);
    LODWORD(v13) = *(_DWORD *)(i + 48);
    sub_CB6200(a2, (unsigned __int8 *)&v13, 4u);
    LODWORD(v13) = (__int64)(*(_QWORD *)(i + 64) - *(_QWORD *)(i + 56)) >> 2;
    sub_CB6200(a2, (unsigned __int8 *)&v13, 4u);
    v7 = *(int **)(i + 64);
    if ( v7 != *(int **)(i + 56) )
    {
      v8 = *(int **)(i + 56);
      do
      {
        v9 = *v8++;
        LODWORD(v13) = v9;
        sub_CB6200(a2, (unsigned __int8 *)&v13, 4u);
      }
      while ( v7 != v8 );
    }
  }
  v10 = v16;
  while ( v10 )
  {
    v11 = v10;
    sub_31152F0(*(_QWORD **)(v10 + 24));
    v12 = *(_QWORD *)(v10 + 56);
    v10 = *(_QWORD *)(v10 + 16);
    if ( v12 )
      j_j___libc_free_0(v12);
    j_j___libc_free_0(v11);
  }
}
