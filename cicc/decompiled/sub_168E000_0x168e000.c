// Function: sub_168E000
// Address: 0x168e000
//
__int64 __fastcall sub_168E000(__int64 a1, _BYTE *a2)
{
  size_t *v3; // r13
  size_t v4; // r15
  const void *v5; // r13
  unsigned int v6; // r8d
  __int64 *v7; // rcx
  __int64 result; // rax
  __int64 v9; // rax
  unsigned int v10; // r8d
  __int64 *v11; // rcx
  __int64 v12; // r12
  void *v13; // rdi
  __int64 *v14; // rdx
  __int64 *v15; // rdx
  __int64 v16; // rax
  void *v17; // rax
  __int64 *v18; // [rsp+0h] [rbp-50h]
  unsigned int v19; // [rsp+8h] [rbp-48h]
  __int64 *v20; // [rsp+8h] [rbp-48h]
  __int64 *v21; // [rsp+10h] [rbp-40h]
  unsigned int v22; // [rsp+10h] [rbp-40h]
  unsigned int v23; // [rsp+18h] [rbp-38h]

  if ( (*a2 & 4) != 0 )
  {
    v3 = (size_t *)*((_QWORD *)a2 - 1);
    v4 = *v3;
    v5 = v3 + 2;
  }
  else
  {
    v4 = 0;
    v5 = 0;
  }
  v6 = sub_16D19C0(a1 + 272, v5, v4);
  v7 = (__int64 *)(*(_QWORD *)(a1 + 272) + 8LL * v6);
  result = *v7;
  if ( *v7 )
  {
    if ( result != -8 )
      goto LABEL_5;
    --*(_DWORD *)(a1 + 288);
  }
  v18 = v7;
  v19 = v6;
  v9 = malloc(v4 + 17);
  v10 = v19;
  v11 = v18;
  v12 = v9;
  if ( !v9 )
  {
    if ( v4 == -17 )
    {
      v16 = malloc(1u);
      v10 = v19;
      v11 = v18;
      if ( v16 )
      {
        v13 = (void *)(v16 + 16);
        v12 = v16;
        goto LABEL_20;
      }
    }
    v20 = v11;
    v22 = v10;
    sub_16BD1C0("Allocation failed");
    v10 = v22;
    v11 = v20;
  }
  v13 = (void *)(v12 + 16);
  if ( v4 + 1 > 1 )
  {
LABEL_20:
    v21 = v11;
    v23 = v10;
    v17 = memcpy(v13, v5, v4);
    v11 = v21;
    v10 = v23;
    v13 = v17;
  }
  *((_BYTE *)v13 + v4) = 0;
  *(_QWORD *)v12 = v4;
  *(_DWORD *)(v12 + 8) = 0;
  *v11 = v12;
  ++*(_DWORD *)(a1 + 284);
  v14 = (__int64 *)(*(_QWORD *)(a1 + 272) + 8LL * (unsigned int)sub_16D1CD0(a1 + 272, v10));
  result = *v14;
  if ( !*v14 || result == -8 )
  {
    v15 = v14 + 1;
    do
    {
      do
        result = *v15++;
      while ( !result );
    }
    while ( result == -8 );
  }
LABEL_5:
  switch ( *(_DWORD *)(result + 8) )
  {
    case 0:
    case 2:
    case 5:
      *(_DWORD *)(result + 8) = 2;
      break;
    case 1:
    case 3:
      *(_DWORD *)(result + 8) = 3;
      break;
    case 6:
      *(_DWORD *)(result + 8) = 4;
      break;
    default:
      return result;
  }
  return result;
}
