// Function: sub_3176AB0
// Address: 0x3176ab0
//
__int64 __fastcall sub_3176AB0(__int64 a1, __int64 a2, unsigned __int8 *a3)
{
  unsigned __int8 *v5; // rax
  __int64 v6; // rdx
  __int64 v7; // r15
  __int64 v8; // rax
  __int64 v9; // rbx
  __int64 *v10; // r8
  int v11; // r10d
  int *v12; // rax
  __int64 v13; // r12
  unsigned __int8 v14; // dl
  __int64 result; // rax
  int v16; // [rsp+8h] [rbp-118h]
  int *v17; // [rsp+8h] [rbp-118h]
  int *v18; // [rsp+8h] [rbp-118h]
  __int64 *v19; // [rsp+10h] [rbp-110h]
  __int64 *v20; // [rsp+10h] [rbp-110h]
  int *v21; // [rsp+18h] [rbp-108h]
  int v22; // [rsp+18h] [rbp-108h]
  int v23; // [rsp+18h] [rbp-108h]
  __int64 v24; // [rsp+28h] [rbp-F8h]
  int v25; // [rsp+50h] [rbp-D0h] BYREF
  int v26; // [rsp+54h] [rbp-CCh]
  unsigned __int64 v27; // [rsp+68h] [rbp-B8h]
  unsigned int v28; // [rsp+70h] [rbp-B0h]
  unsigned __int64 v29; // [rsp+78h] [rbp-A8h]
  unsigned int v30; // [rsp+80h] [rbp-A0h]
  char v31; // [rsp+88h] [rbp-98h]
  int v32; // [rsp+90h] [rbp-90h] BYREF

  v5 = sub_BD3990(a3, a2);
  if ( *v5 )
    return 0;
  if ( !*(_QWORD *)(a1 + 104) )
    sub_4263D6(a3, a2, v6);
  v7 = (__int64)v5;
  v8 = (*(__int64 (__fastcall **)(__int64, unsigned __int8 *))(a1 + 112))(a1 + 88, v5);
  v9 = *(_QWORD *)(a2 + 16);
  v10 = (__int64 *)v8;
  if ( !v9 )
    return 0;
  v11 = 0;
  v12 = &v32;
  do
  {
    v13 = *(_QWORD *)(v9 + 24);
    v14 = *(_BYTE *)v13;
    if ( *(_BYTE *)v13 > 0x1Cu
      && (v14 == 34 || v14 == 85)
      && a2 == *(_QWORD *)(v13 - 32)
      && *(_QWORD *)(v7 + 24) == *(_QWORD *)(v13 + 80) )
    {
      v16 = v11;
      v19 = v10;
      v21 = v12;
      sub_30D6B30((__int64)v12);
      v32 += 100;
      sub_30DEDC0(
        (__int64)&v25,
        (unsigned __int8 *)v13,
        v7,
        v21,
        v19,
        0,
        (__int64)sub_26B9F70,
        a1 + 120,
        (__int64)sub_24258E0,
        a1 + 56,
        0,
        v24,
        0);
      v12 = v21;
      v10 = v19;
      v11 = v16;
      if ( v25 == 0x80000000 )
      {
        v11 = v32 + v16;
      }
      else if ( v25 != 0x7FFFFFFF && v26 - v25 > 0 )
      {
        v11 = v16 + v26 - v25;
      }
      if ( v31 )
      {
        v31 = 0;
        if ( v30 > 0x40 && v29 )
        {
          v17 = v21;
          v22 = v11;
          j_j___libc_free_0_0(v29);
          v12 = v17;
          v10 = v19;
          v11 = v22;
        }
        if ( v28 > 0x40 && v27 )
        {
          v18 = v12;
          v20 = v10;
          v23 = v11;
          j_j___libc_free_0_0(v27);
          v12 = v18;
          v10 = v20;
          v11 = v23;
        }
      }
    }
    v9 = *(_QWORD *)(v9 + 8);
  }
  while ( v9 );
  result = 0;
  if ( v11 >= 0 )
    return (unsigned int)v11;
  return result;
}
