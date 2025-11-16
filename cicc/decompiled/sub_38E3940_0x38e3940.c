// Function: sub_38E3940
// Address: 0x38e3940
//
__int64 __fastcall sub_38E3940(__int64 a1, unsigned __int8 *a2, size_t a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r14
  unsigned int v8; // r15d
  _QWORD *v9; // rcx
  __int64 result; // rax
  __int64 v11; // rax
  _QWORD *v12; // rcx
  _QWORD *v13; // r10
  void *v14; // rdi
  __int64 *v15; // rdx
  __int64 *v16; // rdx
  __int64 v17; // rax
  void *v18; // rax
  _QWORD *v19; // [rsp+8h] [rbp-58h]
  _QWORD *v20; // [rsp+8h] [rbp-58h]
  _QWORD *v21; // [rsp+10h] [rbp-50h]
  _QWORD *v22; // [rsp+10h] [rbp-50h]
  _QWORD *v23; // [rsp+18h] [rbp-48h]

  v5 = a1 + 416;
  v8 = sub_16D19C0(a1 + 416, a2, a3);
  v9 = (_QWORD *)(*(_QWORD *)(a1 + 416) + 8LL * v8);
  result = *v9;
  if ( *v9 )
  {
    if ( result != -8 )
      goto LABEL_3;
    --*(_DWORD *)(a1 + 432);
  }
  v19 = v9;
  v11 = malloc(a3 + 25);
  v12 = v19;
  v13 = (_QWORD *)v11;
  if ( !v11 )
  {
    if ( a3 == -25 )
    {
      v17 = malloc(1u);
      v12 = v19;
      v13 = 0;
      if ( v17 )
      {
        v14 = (void *)(v17 + 24);
        v13 = (_QWORD *)v17;
        goto LABEL_15;
      }
    }
    v20 = v13;
    v22 = v12;
    sub_16BD1C0("Allocation failed", 1u);
    v12 = v22;
    v13 = v20;
  }
  v14 = v13 + 3;
  if ( a3 + 1 > 1 )
  {
LABEL_15:
    v21 = v13;
    v23 = v12;
    v18 = memcpy(v14, a2, a3);
    v13 = v21;
    v12 = v23;
    v14 = v18;
  }
  *((_BYTE *)v14 + a3) = 0;
  *v13 = a3;
  v13[1] = 0;
  v13[2] = 0;
  *v12 = v13;
  ++*(_DWORD *)(a1 + 428);
  v15 = (__int64 *)(*(_QWORD *)(a1 + 416) + 8LL * (unsigned int)sub_16D1CD0(v5, v8));
  result = *v15;
  if ( !*v15 || result == -8 )
  {
    v16 = v15 + 1;
    do
    {
      do
        result = *v16++;
      while ( !result );
    }
    while ( result == -8 );
  }
LABEL_3:
  *(_QWORD *)(result + 8) = a4;
  *(_QWORD *)(result + 16) = a5;
  return result;
}
