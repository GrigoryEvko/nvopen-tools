// Function: sub_38E54C0
// Address: 0x38e54c0
//
__int64 __fastcall sub_38E54C0(__int64 a1, unsigned __int8 *a2, size_t a3, unsigned __int8 *a4, size_t a5)
{
  __int64 v7; // r12
  unsigned int v9; // edx
  __int64 *v10; // rcx
  __int64 v11; // r15
  unsigned int v12; // r14d
  _QWORD *v13; // r10
  __int64 result; // rax
  __int64 v15; // rax
  _QWORD *v16; // r10
  _DWORD *v17; // rcx
  void *v18; // rdi
  __int64 *v19; // rdx
  __int64 v20; // rax
  unsigned int v21; // r10d
  __int64 *v22; // rcx
  __int64 v23; // r11
  void *v24; // rdi
  __int64 *v25; // rax
  __int64 *v26; // rax
  __int64 v27; // rax
  void *v28; // rax
  __int64 v29; // rax
  void *v30; // rax
  __int64 v31; // [rsp+0h] [rbp-60h]
  __int64 *v32; // [rsp+8h] [rbp-58h]
  __int64 v33; // [rsp+8h] [rbp-58h]
  __int64 *v34; // [rsp+8h] [rbp-58h]
  _QWORD *v35; // [rsp+10h] [rbp-50h]
  unsigned int v36; // [rsp+10h] [rbp-50h]
  __int64 *v37; // [rsp+10h] [rbp-50h]
  _QWORD *v38; // [rsp+10h] [rbp-50h]
  unsigned int v39; // [rsp+10h] [rbp-50h]
  _QWORD *srca; // [rsp+18h] [rbp-48h]
  _DWORD *srcb; // [rsp+18h] [rbp-48h]
  unsigned int v43; // [rsp+20h] [rbp-40h]
  _DWORD *v44; // [rsp+20h] [rbp-40h]

  v7 = a1 + 848;
  v9 = sub_16D19C0(a1 + 848, a4, a5);
  v10 = (__int64 *)(*(_QWORD *)(a1 + 848) + 8LL * v9);
  v11 = *v10;
  if ( *v10 )
  {
    if ( v11 != -8 )
      goto LABEL_3;
    --*(_DWORD *)(a1 + 864);
  }
  v32 = v10;
  v36 = v9;
  v20 = malloc(a5 + 17);
  v21 = v36;
  v22 = v32;
  v23 = v20;
  if ( !v20 )
  {
    if ( a5 == -17 )
    {
      v27 = malloc(1u);
      v21 = v36;
      v22 = v32;
      v23 = 0;
      if ( v27 )
      {
        v24 = (void *)(v27 + 16);
        v23 = v27;
        goto LABEL_25;
      }
    }
    v31 = v23;
    v34 = v22;
    v39 = v21;
    sub_16BD1C0("Allocation failed", 1u);
    v21 = v39;
    v22 = v34;
    v23 = v31;
  }
  v24 = (void *)(v23 + 16);
  if ( a5 + 1 > 1 )
  {
LABEL_25:
    v33 = v23;
    v37 = v22;
    v43 = v21;
    v28 = memcpy(v24, a4, a5);
    v23 = v33;
    v22 = v37;
    v21 = v43;
    v24 = v28;
  }
  *((_BYTE *)v24 + a5) = 0;
  *(_QWORD *)v23 = a5;
  *(_DWORD *)(v23 + 8) = 0;
  *v22 = v23;
  ++*(_DWORD *)(a1 + 860);
  v25 = (__int64 *)(*(_QWORD *)(a1 + 848) + 8LL * (unsigned int)sub_16D1CD0(v7, v21));
  v11 = *v25;
  if ( *v25 == -8 || !v11 )
  {
    v26 = v25 + 1;
    do
    {
      do
        v11 = *v26++;
      while ( !v11 );
    }
    while ( v11 == -8 );
  }
LABEL_3:
  v12 = sub_16D19C0(v7, a2, a3);
  v13 = (_QWORD *)(*(_QWORD *)(a1 + 848) + 8LL * v12);
  result = *v13;
  if ( *v13 )
  {
    if ( result != -8 )
      goto LABEL_5;
    --*(_DWORD *)(a1 + 864);
  }
  v35 = v13;
  v15 = malloc(a3 + 17);
  v16 = v35;
  v17 = (_DWORD *)v15;
  if ( !v15 )
  {
    if ( a3 == -17 )
    {
      v29 = malloc(1u);
      v17 = 0;
      v16 = v35;
      if ( v29 )
      {
        v18 = (void *)(v29 + 16);
        v17 = (_DWORD *)v29;
        goto LABEL_28;
      }
    }
    v38 = v16;
    srcb = v17;
    sub_16BD1C0("Allocation failed", 1u);
    v17 = srcb;
    v16 = v38;
  }
  v18 = v17 + 4;
  if ( a3 + 1 > 1 )
  {
LABEL_28:
    srca = v16;
    v44 = v17;
    v30 = memcpy(v18, a2, a3);
    v16 = srca;
    v17 = v44;
    v18 = v30;
  }
  *((_BYTE *)v18 + a3) = 0;
  *(_QWORD *)v17 = a3;
  v17[2] = 0;
  *v16 = v17;
  ++*(_DWORD *)(a1 + 860);
  v19 = (__int64 *)(*(_QWORD *)(a1 + 848) + 8LL * (unsigned int)sub_16D1CD0(v7, v12));
  result = *v19;
  if ( !*v19 || result == -8 )
  {
    do
    {
      do
      {
        result = v19[1];
        ++v19;
      }
      while ( result == -8 );
    }
    while ( !result );
  }
LABEL_5:
  *(_DWORD *)(result + 8) = *(_DWORD *)(v11 + 8);
  return result;
}
