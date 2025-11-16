// Function: sub_19C1B10
// Address: 0x19c1b10
//
unsigned __int64 __fastcall sub_19C1B10(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 *v2; // rax
  _QWORD *v3; // rax
  __int64 v4; // rdx
  __int64 v5; // r13
  _BOOL4 v6; // r9d
  __int64 v7; // rax
  unsigned __int64 v8; // r12
  char v9; // al
  __int64 v10; // rbx
  __int64 v11; // rdi
  unsigned __int64 v13; // [rsp+0h] [rbp-80h]
  _BOOL4 v14; // [rsp+0h] [rbp-80h]
  unsigned __int64 v15; // [rsp+18h] [rbp-68h] BYREF
  _BYTE v16[8]; // [rsp+20h] [rbp-60h] BYREF
  int v17; // [rsp+28h] [rbp-58h] BYREF
  __int64 v18; // [rsp+30h] [rbp-50h]
  int *v19; // [rsp+38h] [rbp-48h]
  int *v20; // [rsp+40h] [rbp-40h]
  __int64 v21; // [rsp+48h] [rbp-38h]

  v2 = *(unsigned __int64 **)(a1 + 32);
  v19 = &v17;
  v17 = 0;
  v18 = 0;
  v20 = &v17;
  v21 = 0;
  v15 = *v2;
  v13 = v15;
  v3 = sub_19C18C0((__int64)v16, &v15);
  if ( v4 )
  {
    v5 = v4;
    v6 = v3 || (int *)v4 == &v17 || v13 < *(_QWORD *)(v4 + 32);
    v14 = v6;
    v7 = sub_22077B0(40);
    *(_QWORD *)(v7 + 32) = v15;
    sub_220F040(v14, v7, v5, &v17);
    ++v21;
  }
  v15 = 0;
  v8 = 0;
  v9 = sub_19C1960(a1, a2, &v15, (__int64)v16);
  v10 = v18;
  if ( v9 )
    v8 = v15;
  if ( v18 )
  {
    do
    {
      sub_19C0900(*(_QWORD *)(v10 + 24));
      v11 = v10;
      v10 = *(_QWORD *)(v10 + 16);
      j_j___libc_free_0(v11, 40);
    }
    while ( v10 );
  }
  return v8;
}
