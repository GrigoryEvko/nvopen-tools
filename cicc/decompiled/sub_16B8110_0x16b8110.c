// Function: sub_16B8110
// Address: 0x16b8110
//
__int64 __fastcall sub_16B8110(__int64 a1, const void *a2, size_t a3, _QWORD *a4)
{
  unsigned int v6; // r15d
  _QWORD *v7; // rbx
  _QWORD *v9; // rcx
  void *v10; // rdi
  __int64 *v11; // rcx
  __int64 *v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  void *v15; // rax
  _QWORD *v16; // [rsp+8h] [rbp-48h]
  _QWORD *v17; // [rsp+10h] [rbp-40h]

  v6 = sub_16D19C0(a1, a2, a3);
  v7 = (_QWORD *)(*(_QWORD *)a1 + 8LL * v6);
  if ( *v7 )
  {
    if ( *v7 != -8 )
      return *(_QWORD *)a1 + 8LL * v6;
    --*(_DWORD *)(a1 + 16);
  }
  v9 = (_QWORD *)malloc(a3 + 17);
  if ( v9 )
  {
LABEL_6:
    v10 = v9 + 2;
    if ( a3 + 1 <= 1 )
      goto LABEL_7;
    goto LABEL_15;
  }
  if ( a3 != -17 || (v14 = malloc(1u), v9 = 0, !v14) )
  {
    v16 = v9;
    sub_16BD1C0("Allocation failed");
    v9 = v16;
    goto LABEL_6;
  }
  v10 = (void *)(v14 + 16);
  v9 = (_QWORD *)v14;
LABEL_15:
  v17 = v9;
  v15 = memcpy(v10, a2, a3);
  v9 = v17;
  v10 = v15;
LABEL_7:
  *((_BYTE *)v10 + a3) = 0;
  *v9 = a3;
  v9[1] = *a4;
  *v7 = v9;
  ++*(_DWORD *)(a1 + 12);
  v11 = (__int64 *)(*(_QWORD *)a1 + 8LL * (unsigned int)sub_16D1CD0(a1, v6));
  if ( *v11 == -8 || !*v11 )
  {
    v12 = v11 + 1;
    do
    {
      do
      {
        v13 = *v12;
        v11 = v12++;
      }
      while ( !v13 );
    }
    while ( v13 == -8 );
  }
  return (__int64)v11;
}
