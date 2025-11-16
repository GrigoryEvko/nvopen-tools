// Function: sub_39417C0
// Address: 0x39417c0
//
__int64 __fastcall sub_39417C0(
        __int64 a1,
        unsigned int a2,
        unsigned int a3,
        unsigned __int8 *a4,
        size_t a5,
        unsigned __int64 a6,
        unsigned __int64 a7)
{
  __int64 v8; // rax
  __int64 v9; // r13
  unsigned int v10; // edx
  _QWORD *v11; // rcx
  __int64 v12; // rbx
  __int64 v14; // rax
  unsigned int v15; // r10d
  _QWORD *v16; // rcx
  _QWORD *v17; // r11
  void *v18; // rdi
  __int64 *v19; // rax
  __int64 v20; // rax
  void *v21; // rax
  _QWORD *v22; // [rsp+0h] [rbp-70h]
  _QWORD *v23; // [rsp+8h] [rbp-68h]
  _QWORD *v24; // [rsp+8h] [rbp-68h]
  unsigned int v25; // [rsp+8h] [rbp-68h]
  unsigned int v26; // [rsp+10h] [rbp-60h]
  unsigned int v27; // [rsp+10h] [rbp-60h]
  _QWORD *v28; // [rsp+10h] [rbp-60h]
  _QWORD *v29; // [rsp+18h] [rbp-58h]
  __int64 *v32; // [rsp+30h] [rbp-40h] BYREF
  _DWORD v33[14]; // [rsp+38h] [rbp-38h] BYREF

  v8 = *(_QWORD *)(a1 + 48);
  v9 = a1 + 40;
  v33[0] = a2;
  v33[1] = a3;
  if ( !v8 )
  {
LABEL_10:
    v32 = (__int64 *)v33;
    v9 = sub_39416E0((_QWORD *)(a1 + 32), v9, &v32);
    goto LABEL_11;
  }
  do
  {
    while ( 1 )
    {
      if ( a2 > *(_DWORD *)(v8 + 32) )
      {
        v8 = *(_QWORD *)(v8 + 24);
        goto LABEL_7;
      }
      if ( a2 == *(_DWORD *)(v8 + 32) && a3 > *(_DWORD *)(v8 + 36) )
        break;
      v9 = v8;
      v8 = *(_QWORD *)(v8 + 16);
      if ( !v8 )
        goto LABEL_8;
    }
    v8 = *(_QWORD *)(v8 + 24);
LABEL_7:
    ;
  }
  while ( v8 );
LABEL_8:
  if ( a1 + 40 == v9 || a2 < *(_DWORD *)(v9 + 32) || a2 == *(_DWORD *)(v9 + 32) && a3 < *(_DWORD *)(v9 + 36) )
    goto LABEL_10;
LABEL_11:
  v10 = sub_16D19C0(v9 + 48, a4, a5);
  v11 = (_QWORD *)(*(_QWORD *)(v9 + 48) + 8LL * v10);
  v12 = *v11;
  if ( *v11 )
  {
    if ( v12 != -8 )
      goto LABEL_13;
    --*(_DWORD *)(v9 + 64);
  }
  v23 = v11;
  v26 = v10;
  v14 = malloc(a5 + 17);
  v15 = v26;
  v16 = v23;
  v17 = (_QWORD *)v14;
  if ( !v14 )
  {
    if ( a5 != -17 || (v20 = malloc(1u), v17 = 0, v15 = v26, v16 = v23, !v20) )
    {
      v22 = v16;
      v25 = v15;
      v28 = v17;
      sub_16BD1C0("Allocation failed", 1u);
      v17 = v28;
      v15 = v25;
      v16 = v22;
      goto LABEL_18;
    }
    v18 = (void *)(v20 + 16);
    v17 = (_QWORD *)v20;
    goto LABEL_29;
  }
LABEL_18:
  v18 = v17 + 2;
  if ( a5 + 1 > 1 )
  {
LABEL_29:
    v24 = v16;
    v27 = v15;
    v29 = v17;
    v21 = memcpy(v18, a4, a5);
    v16 = v24;
    v15 = v27;
    v17 = v29;
    v18 = v21;
  }
  *((_BYTE *)v18 + a5) = 0;
  *v17 = a5;
  v17[1] = 0;
  *v16 = v17;
  ++*(_DWORD *)(v9 + 60);
  v19 = (__int64 *)(*(_QWORD *)(v9 + 48) + 8LL * (unsigned int)sub_16D1CD0(v9 + 48, v15));
  v12 = *v19;
  if ( *v19 == -8 || !v12 )
  {
    do
    {
      do
      {
        v12 = v19[1];
        ++v19;
      }
      while ( !v12 );
    }
    while ( v12 == -8 );
  }
LABEL_13:
  *(_QWORD *)(v12 + 8) = sub_393FEE0(a6, a7, *(_QWORD *)(v12 + 8), (bool *)&v32);
  return (_BYTE)v32 != 0 ? 0xA : 0;
}
