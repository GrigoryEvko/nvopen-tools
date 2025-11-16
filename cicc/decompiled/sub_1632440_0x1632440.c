// Function: sub_1632440
// Address: 0x1632440
//
__int64 __fastcall sub_1632440(__int64 a1, const void *a2, size_t a3)
{
  __int64 v5; // r14
  __int64 v6; // rdx
  _QWORD *v7; // r9
  __int64 v8; // rbx
  __int64 result; // rax
  __int64 v10; // rax
  unsigned int v11; // r8d
  _QWORD *v12; // r9
  _QWORD *v13; // rcx
  void *v14; // rdi
  __int64 *v15; // rax
  __int64 *v16; // rax
  __int64 v17; // rax
  __int64 v18; // r12
  __int64 *v19; // rax
  __int64 v20; // rcx
  __int64 v21; // rax
  void *v22; // rax
  _QWORD *v23; // [rsp+0h] [rbp-80h]
  _QWORD *v24; // [rsp+8h] [rbp-78h]
  _QWORD *v25; // [rsp+8h] [rbp-78h]
  _QWORD *v26; // [rsp+8h] [rbp-78h]
  unsigned int v27; // [rsp+10h] [rbp-70h]
  _QWORD *v28; // [rsp+10h] [rbp-70h]
  unsigned int v29; // [rsp+10h] [rbp-70h]
  unsigned int v30; // [rsp+18h] [rbp-68h]
  _QWORD v31[2]; // [rsp+20h] [rbp-60h] BYREF
  _QWORD *v32; // [rsp+30h] [rbp-50h] BYREF
  __int16 v33; // [rsp+40h] [rbp-40h]

  v5 = *(_QWORD *)(a1 + 272);
  v31[1] = a3;
  v31[0] = a2;
  v6 = (unsigned int)sub_16D19C0(v5, a2, a3);
  v7 = (_QWORD *)(*(_QWORD *)v5 + 8 * v6);
  v8 = *v7;
  if ( *v7 )
  {
    if ( v8 != -8 )
      goto LABEL_3;
    --*(_DWORD *)(v5 + 16);
  }
  v24 = v7;
  v27 = v6;
  v10 = malloc(a3 + 17);
  v11 = v27;
  v12 = v24;
  v13 = (_QWORD *)v10;
  if ( !v10 )
  {
    if ( a3 == -17 )
    {
      v21 = malloc(1u);
      v11 = v27;
      v12 = v24;
      v13 = 0;
      if ( v21 )
      {
        v14 = (void *)(v21 + 16);
        v13 = (_QWORD *)v21;
        goto LABEL_19;
      }
    }
    v23 = v13;
    v26 = v12;
    v29 = v11;
    sub_16BD1C0("Allocation failed");
    v11 = v29;
    v12 = v26;
    v13 = v23;
  }
  v14 = v13 + 2;
  if ( a3 + 1 > 1 )
  {
LABEL_19:
    v25 = v13;
    v28 = v12;
    v30 = v11;
    v22 = memcpy(v14, a2, a3);
    v13 = v25;
    v12 = v28;
    v11 = v30;
    v14 = v22;
  }
  *((_BYTE *)v14 + a3) = 0;
  *v13 = a3;
  v13[1] = 0;
  *v12 = v13;
  ++*(_DWORD *)(v5 + 12);
  v15 = (__int64 *)(*(_QWORD *)v5 + 8LL * (unsigned int)sub_16D1CD0(v5, v11));
  v8 = *v15;
  if ( *v15 && v8 != -8 )
  {
LABEL_3:
    result = *(_QWORD *)(v8 + 8);
    if ( result )
      return result;
    goto LABEL_14;
  }
  v16 = v15 + 1;
  do
  {
    do
      v8 = *v16++;
    while ( !v8 );
  }
  while ( v8 == -8 );
  result = *(_QWORD *)(v8 + 8);
  if ( !result )
  {
LABEL_14:
    v33 = 261;
    v32 = v31;
    v17 = sub_22077B0(64);
    v18 = v17;
    if ( v17 )
      sub_161F4C0(v17, (__int64)&v32);
    *(_QWORD *)(v8 + 8) = v18;
    *(_QWORD *)(v18 + 48) = a1;
    v19 = *(__int64 **)(v8 + 8);
    v20 = *(_QWORD *)(a1 + 72);
    v19[1] = a1 + 72;
    v20 &= 0xFFFFFFFFFFFFFFF8LL;
    *v19 = v20 | *v19 & 7;
    *(_QWORD *)(v20 + 8) = v19;
    *(_QWORD *)(a1 + 72) = *(_QWORD *)(a1 + 72) & 7LL | (unsigned __int64)v19;
    return *(_QWORD *)(v8 + 8);
  }
  return result;
}
