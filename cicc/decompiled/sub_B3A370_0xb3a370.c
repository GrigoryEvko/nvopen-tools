// Function: sub_B3A370
// Address: 0xb3a370
//
__int64 __fastcall sub_B3A370(__int64 a1, __int64 a2, __m128i a3)
{
  char v4; // r13
  __int64 v5; // rdx
  char v6; // al
  __int64 j; // r12
  __int64 v9; // rdi
  __int64 i; // r15
  char v11; // cl
  __int64 v12; // r14
  __int64 v13; // rax
  __int64 v14; // rdx
  char v15; // al
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rdi
  _BYTE *v19; // rax
  __int64 v20; // r15
  __int64 v21; // rdi
  char v22; // al
  __int64 v23; // rdi
  _BYTE *v24; // rax
  __int64 k; // r12
  __int64 v26; // rdi
  __int64 v27; // r15
  __int64 v28; // rdi
  char v29; // [rsp+6h] [rbp-3Ah]
  char v30; // [rsp+7h] [rbp-39h]

  v30 = *(_BYTE *)(a2 + 872);
  if ( unk_4F81788 )
  {
    if ( !*(_BYTE *)(a2 + 872) )
    {
      v20 = *(_QWORD *)(a2 + 32);
      if ( v20 != a2 + 24 )
      {
        do
        {
          v21 = v20 - 56;
          if ( !v20 )
            v21 = 0;
          sub_B2B950(v21);
          v20 = *(_QWORD *)(v20 + 8);
        }
        while ( a2 + 24 != v20 );
        *(_BYTE *)(a2 + 872) = 1;
        v22 = unk_4F81788;
        goto LABEL_35;
      }
      *(_BYTE *)(a2 + 872) = 1;
    }
LABEL_29:
    sub_BA8950(a2);
    goto LABEL_3;
  }
  if ( !v30 )
    goto LABEL_3;
  v27 = *(_QWORD *)(a2 + 32);
  if ( v27 == a2 + 24 )
  {
    *(_BYTE *)(a2 + 872) = 0;
    goto LABEL_3;
  }
  do
  {
    v28 = v27 - 56;
    if ( !v27 )
      v28 = 0;
    sub_B2B9A0(v28);
    v27 = *(_QWORD *)(v27 + 8);
  }
  while ( a2 + 24 != v27 );
  *(_BYTE *)(a2 + 872) = 0;
  v22 = unk_4F81788;
LABEL_35:
  if ( v22 )
    goto LABEL_29;
LABEL_3:
  v4 = sub_BC63A0("*", 1);
  if ( v4 )
  {
    v5 = *(_QWORD *)(a1 + 192);
    if ( v5 )
    {
      v23 = sub_CB6200(*(_QWORD *)(a1 + 176), *(_QWORD *)(a1 + 184), v5);
      v24 = *(_BYTE **)(v23 + 32);
      if ( *(_BYTE **)(v23 + 24) == v24 )
      {
        sub_CB6200(v23, "\n", 1);
      }
      else
      {
        *v24 = 10;
        ++*(_QWORD *)(v23 + 32);
      }
    }
    sub_A69980((__int64 (__fastcall **)())a2, *(_QWORD *)(a1 + 176), 0, *(_BYTE *)(a1 + 216), 0, a3);
  }
  else
  {
    for ( i = *(_QWORD *)(a2 + 32); a2 + 24 != i; i = *(_QWORD *)(i + 8) )
    {
      v12 = 0;
      if ( i )
        v12 = i - 56;
      v13 = sub_BD5D20(v12);
      v15 = sub_BC63A0(v13, v14);
      if ( v15 )
      {
        if ( !v4 )
        {
          v16 = *(_QWORD *)(a1 + 192);
          if ( v16 )
          {
            v29 = v15;
            v17 = sub_CB6200(*(_QWORD *)(a1 + 176), *(_QWORD *)(a1 + 184), v16);
            v11 = v29;
            v18 = v17;
            v19 = *(_BYTE **)(v17 + 32);
            if ( *(_BYTE **)(v18 + 24) == v19 )
            {
              sub_CB6200(v18, "\n", 1);
              v11 = v29;
            }
            else
            {
              *v19 = 10;
              ++*(_QWORD *)(v18 + 32);
            }
            v4 = v11;
          }
        }
        sub_A68C30(v12, *(_QWORD *)(a1 + 176), 0, 0, 0);
      }
    }
  }
  v6 = *(_BYTE *)(a2 + 872);
  if ( v30 )
  {
    if ( !v6 )
    {
      for ( j = *(_QWORD *)(a2 + 32); a2 + 24 != j; j = *(_QWORD *)(j + 8) )
      {
        v9 = j - 56;
        if ( !j )
          v9 = 0;
        sub_B2B950(v9);
      }
      *(_BYTE *)(a2 + 872) = 1;
    }
  }
  else if ( v6 )
  {
    for ( k = *(_QWORD *)(a2 + 32); a2 + 24 != k; k = *(_QWORD *)(k + 8) )
    {
      v26 = k - 56;
      if ( !k )
        v26 = 0;
      sub_B2B9A0(v26);
    }
    *(_BYTE *)(a2 + 872) = 0;
  }
  return 0;
}
