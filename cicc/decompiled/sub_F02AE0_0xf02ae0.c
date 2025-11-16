// Function: sub_F02AE0
// Address: 0xf02ae0
//
__int64 __fastcall sub_F02AE0(__int64 a1, __int64 a2, const void *a3, size_t a4)
{
  int v6; // eax
  unsigned int v7; // r9d
  unsigned __int64 *v8; // r10
  _QWORD *v9; // rax
  __int64 v10; // rdx
  int v11; // ecx
  __int64 v13; // rax
  unsigned __int64 v14; // r15
  size_t v15; // rdx
  _BYTE *v16; // rcx
  __int64 v17; // rax
  _QWORD *v18; // rdx
  _BYTE *v19; // rax
  __int64 v20; // rax
  unsigned __int64 *v21; // [rsp+0h] [rbp-60h]
  unsigned __int64 *v22; // [rsp+0h] [rbp-60h]
  unsigned int v23; // [rsp+8h] [rbp-58h]
  unsigned int v24; // [rsp+8h] [rbp-58h]
  int v25; // [rsp+Ch] [rbp-54h]

  v25 = *(_DWORD *)(a2 + 12);
  v6 = sub_C92610();
  v7 = sub_C92740(a2, a3, a4, v6);
  v8 = (unsigned __int64 *)(*(_QWORD *)a2 + 8LL * v7);
  v9 = (_QWORD *)*v8;
  if ( *v8 )
  {
    if ( v9 != (_QWORD *)-8LL )
      goto LABEL_3;
    --*(_DWORD *)(a2 + 16);
  }
  v13 = *(_QWORD *)(a2 + 24);
  *(_QWORD *)(a2 + 104) += a4 + 17;
  v14 = (v13 + 7) & 0xFFFFFFFFFFFFFFF8LL;
  v15 = a4 + 17 + v14;
  if ( *(_QWORD *)(a2 + 32) < v15 || !v13 )
  {
    v22 = v8;
    v24 = v7;
    v20 = sub_9D1E70(a2 + 24, a4 + 17, a4 + 17, 3);
    v7 = v24;
    v8 = v22;
    v14 = v20;
    v16 = (_BYTE *)(v20 + 16);
    if ( !a4 )
    {
      *(_BYTE *)(v20 + 16) = 0;
      goto LABEL_9;
    }
LABEL_14:
    v21 = v8;
    v23 = v7;
    v19 = memcpy(v16, a3, a4);
    v8 = v21;
    v7 = v23;
    v19[a4] = 0;
    if ( !v14 )
      goto LABEL_10;
    goto LABEL_9;
  }
  *(_QWORD *)(a2 + 24) = v15;
  v16 = (_BYTE *)(v14 + 16);
  if ( a4 )
    goto LABEL_14;
  *v16 = 0;
  if ( v14 )
  {
LABEL_9:
    *(_QWORD *)v14 = a4;
    *(_DWORD *)(v14 + 8) = v25;
  }
LABEL_10:
  *v8 = v14;
  ++*(_DWORD *)(a2 + 12);
  v17 = *(_QWORD *)a2 + 8LL * (unsigned int)sub_C929D0((__int64 *)a2, v7);
  v18 = *(_QWORD **)v17;
  if ( *(_QWORD *)v17 != -8 )
    goto LABEL_12;
  do
  {
    do
    {
      v18 = *(_QWORD **)(v17 + 8);
      v17 += 8;
    }
    while ( v18 == (_QWORD *)-8LL );
LABEL_12:
    ;
  }
  while ( !v18 );
  *(_QWORD *)(a2 + 120) += *v18 + 1LL;
  v9 = *(_QWORD **)v17;
LABEL_3:
  v10 = *v9;
  v11 = *((_DWORD *)v9 + 2);
  *(_QWORD *)(a1 + 8) = v9 + 2;
  *(_DWORD *)a1 = v11;
  *(_QWORD *)(a1 + 16) = v10;
  return a1;
}
