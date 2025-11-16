// Function: sub_1633B90
// Address: 0x1633b90
//
__int64 __fastcall sub_1633B90(__int64 a1, void *a2, size_t a3)
{
  __int64 v3; // r13
  void *v6; // r15
  size_t v7; // r12
  __int64 v8; // rax
  _QWORD *v10; // rcx
  void *v11; // rdi
  __int64 *v12; // rdx
  __int64 *v13; // rdx
  __int64 v14; // rax
  void *v15; // rax
  _QWORD *v16; // [rsp+0h] [rbp-80h]
  _QWORD *v17; // [rsp+8h] [rbp-78h]
  _QWORD *v18; // [rsp+8h] [rbp-78h]
  __int64 *v19; // [rsp+10h] [rbp-70h]
  unsigned int v20; // [rsp+1Ch] [rbp-64h]
  _BYTE v21[16]; // [rsp+20h] [rbp-60h] BYREF
  void *src; // [rsp+30h] [rbp-50h]
  size_t n; // [rsp+38h] [rbp-48h]
  _BYTE v24[64]; // [rsp+40h] [rbp-40h] BYREF

  v3 = a1 + 128;
  sub_1580C60((__int64)v21);
  src = a2;
  n = a3;
  sub_1580C50((__int64)v24, (__int64)v21);
  v6 = src;
  v7 = n;
  v20 = sub_16D19C0(a1 + 128, src, n);
  v19 = (__int64 *)(*(_QWORD *)(a1 + 128) + 8LL * v20);
  v8 = *v19;
  if ( *v19 )
  {
    if ( v8 != -8 )
      goto LABEL_3;
    --*(_DWORD *)(a1 + 144);
  }
  v10 = (_QWORD *)malloc(v7 + 25);
  if ( !v10 )
  {
    if ( v7 == -25 )
    {
      v14 = malloc(1u);
      v10 = 0;
      if ( v14 )
      {
        v11 = (void *)(v14 + 24);
        v10 = (_QWORD *)v14;
        goto LABEL_15;
      }
    }
    v16 = v10;
    sub_16BD1C0("Allocation failed");
    v10 = v16;
  }
  v11 = v10 + 3;
  if ( v7 + 1 > 1 )
  {
LABEL_15:
    v18 = v10;
    v15 = memcpy(v11, v6, v7);
    v10 = v18;
    v11 = v15;
  }
  *((_BYTE *)v11 + v7) = 0;
  *v10 = v7;
  v17 = v10;
  sub_1580C50((__int64)(v10 + 1), (__int64)v24);
  *v19 = (__int64)v17;
  ++*(_DWORD *)(a1 + 140);
  v12 = (__int64 *)(*(_QWORD *)(a1 + 128) + 8LL * (unsigned int)sub_16D1CD0(v3, v20));
  v8 = *v12;
  if ( !*v12 || v8 == -8 )
  {
    v13 = v12 + 1;
    do
    {
      do
        v8 = *v13++;
      while ( !v8 );
    }
    while ( v8 == -8 );
  }
LABEL_3:
  *(_QWORD *)(v8 + 8) = v8;
  return v8 + 8;
}
