// Function: sub_16052C0
// Address: 0x16052c0
//
__int64 __fastcall sub_16052C0(__int64 a1, const void *a2, size_t a3)
{
  __int64 v4; // r13
  unsigned int v6; // r15d
  __int64 *v7; // r9
  __int64 v8; // r8
  __int64 v10; // rax
  __int64 *v11; // r9
  __int64 v12; // rcx
  void *v13; // rdi
  __int64 *v14; // rax
  __int64 *v15; // rax
  __int64 v16; // rax
  void *v17; // rax
  __int64 *v18; // [rsp+0h] [rbp-50h]
  __int64 v19; // [rsp+0h] [rbp-50h]
  __int64 v20; // [rsp+8h] [rbp-48h]
  __int64 *v21; // [rsp+8h] [rbp-48h]
  __int64 *v22; // [rsp+10h] [rbp-40h]
  int v23; // [rsp+1Ch] [rbp-34h]

  v4 = a1 + 2864;
  v23 = *(_DWORD *)(a1 + 2876);
  v6 = sub_16D19C0(a1 + 2864, a2, a3);
  v7 = (__int64 *)(*(_QWORD *)(a1 + 2864) + 8LL * v6);
  v8 = *v7;
  if ( *v7 )
  {
    if ( v8 != -8 )
      return v8;
    --*(_DWORD *)(a1 + 2880);
  }
  v18 = v7;
  v10 = malloc(a3 + 17);
  v11 = v18;
  v12 = v10;
  if ( !v10 )
  {
    if ( a3 == -17 )
    {
      v16 = malloc(1u);
      v11 = v18;
      v12 = 0;
      if ( v16 )
      {
        v13 = (void *)(v16 + 16);
        v12 = v16;
        goto LABEL_15;
      }
    }
    v19 = v12;
    v21 = v11;
    sub_16BD1C0("Allocation failed");
    v11 = v21;
    v12 = v19;
  }
  v13 = (void *)(v12 + 16);
  if ( a3 + 1 > 1 )
  {
LABEL_15:
    v20 = v12;
    v22 = v11;
    v17 = memcpy(v13, a2, a3);
    v12 = v20;
    v11 = v22;
    v13 = v17;
  }
  *((_BYTE *)v13 + a3) = 0;
  *(_QWORD *)v12 = a3;
  *(_DWORD *)(v12 + 8) = v23;
  *v11 = v12;
  ++*(_DWORD *)(a1 + 2876);
  v14 = (__int64 *)(*(_QWORD *)(a1 + 2864) + 8LL * (unsigned int)sub_16D1CD0(v4, v6));
  v8 = *v14;
  if ( !*v14 || v8 == -8 )
  {
    v15 = v14 + 1;
    do
    {
      do
        v8 = *v15++;
      while ( !v8 );
    }
    while ( v8 == -8 );
  }
  return v8;
}
