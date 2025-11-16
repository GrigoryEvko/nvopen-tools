// Function: sub_B9B140
// Address: 0xb9b140
//
__int64 __fastcall sub_B9B140(__int64 *a1, const void *a2, size_t a3)
{
  __int64 v4; // rbx
  unsigned int v5; // eax
  unsigned int v6; // r8d
  _QWORD *v7; // r9
  __int64 v8; // rax
  __int64 v10; // rax
  _QWORD *v11; // r13
  char *v12; // rdx
  _BYTE *v13; // rcx
  __int64 *v14; // rax
  __int64 v15; // rdx
  _BYTE *v16; // rax
  __int64 v17; // rax
  _QWORD *v18; // [rsp+0h] [rbp-40h]
  _QWORD *v19; // [rsp+0h] [rbp-40h]
  unsigned int v20; // [rsp+Ch] [rbp-34h]
  unsigned int v21; // [rsp+Ch] [rbp-34h]

  v4 = *a1;
  v5 = sub_C92610(a2, a3);
  v6 = sub_C92740(v4 + 448, a2, a3, v5);
  v7 = (_QWORD *)(*(_QWORD *)(v4 + 448) + 8LL * v6);
  v8 = *v7;
  if ( *v7 )
  {
    if ( v8 != -8 )
      return v8 + 8;
    --*(_DWORD *)(v4 + 464);
  }
  v10 = *(_QWORD *)(v4 + 472);
  *(_QWORD *)(v4 + 552) += a3 + 25;
  v11 = (_QWORD *)((v10 + 7) & 0xFFFFFFFFFFFFFFF8LL);
  v12 = (char *)v11 + a3 + 25;
  if ( *(_QWORD *)(v4 + 480) >= (unsigned __int64)v12 && v10 )
  {
    *(_QWORD *)(v4 + 472) = v12;
    v13 = v11 + 3;
    if ( !a3 )
    {
      *v13 = 0;
      if ( !v11 )
        goto LABEL_10;
      goto LABEL_9;
    }
  }
  else
  {
    v19 = v7;
    v21 = v6;
    v17 = sub_9D1E70(v4 + 472, a3 + 25, a3 + 25, 3);
    v6 = v21;
    v7 = v19;
    v11 = (_QWORD *)v17;
    v13 = (_BYTE *)(v17 + 24);
    if ( !a3 )
    {
      *(_BYTE *)(v17 + 24) = 0;
      goto LABEL_9;
    }
  }
  v18 = v7;
  v20 = v6;
  v16 = memcpy(v13, a2, a3);
  v7 = v18;
  v6 = v20;
  v16[a3] = 0;
  if ( v11 )
  {
LABEL_9:
    *v11 = a3;
    v11[1] = 0;
    v11[2] = 0;
  }
LABEL_10:
  *v7 = v11;
  ++*(_DWORD *)(v4 + 460);
  v14 = (__int64 *)(*(_QWORD *)(v4 + 448) + 8LL * (unsigned int)sub_C929D0(v4 + 448, v6));
  v15 = *v14;
  if ( !*v14 || v15 == -8 )
  {
    do
    {
      do
      {
        v15 = v14[1];
        ++v14;
      }
      while ( v15 == -8 );
    }
    while ( !v15 );
  }
  *(_QWORD *)(v15 + 16) = v15;
  return v15 + 8;
}
