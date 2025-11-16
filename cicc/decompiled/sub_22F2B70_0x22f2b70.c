// Function: sub_22F2B70
// Address: 0x22f2b70
//
__int64 __fastcall sub_22F2B70(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __m128i a5)
{
  size_t v7; // rdx
  char v8; // al
  __int64 j; // rbx
  __int64 v11; // rdi
  __int64 i; // r15
  bool v13; // cl
  __int64 v14; // r14
  char *v15; // rax
  __int64 v16; // rdx
  bool v17; // al
  size_t v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rdi
  _BYTE *v21; // rax
  __int64 v22; // r15
  __int64 v23; // r8
  __int64 v24; // rdi
  _BYTE *v25; // rax
  __int64 v26; // r15
  __int64 v27; // r14
  __int64 v28; // rdi
  char v29; // al
  __int64 k; // rbx
  __int64 v31; // rdi
  __int64 v32; // r15
  __int64 v33; // r14
  __int64 v34; // rdi
  int v35; // eax
  unsigned int v36; // eax
  __int64 *v37; // rdx
  __int64 v38; // rax
  __m128i v39; // xmm0
  int v40; // ecx
  __int64 *v42; // [rsp+8h] [rbp-78h]
  char v43; // [rsp+13h] [rbp-6Dh]
  bool v44; // [rsp+14h] [rbp-6Ch]
  bool v45; // [rsp+14h] [rbp-6Ch]
  unsigned int v46; // [rsp+14h] [rbp-6Ch]
  __m128i v47; // [rsp+30h] [rbp-50h] BYREF
  int v48; // [rsp+40h] [rbp-40h]

  v43 = *(_BYTE *)(a3 + 872);
  if ( unk_4F81788 )
  {
    if ( !*(_BYTE *)(a3 + 872) )
    {
      v32 = *(_QWORD *)(a3 + 32);
      v33 = a3 + 24;
      if ( v32 != a3 + 24 )
      {
        do
        {
          v34 = v32 - 56;
          if ( !v32 )
            v34 = 0;
          sub_B2B950(v34);
          v32 = *(_QWORD *)(v32 + 8);
        }
        while ( v33 != v32 );
        *(_BYTE *)(a3 + 872) = 1;
        v29 = unk_4F81788;
        goto LABEL_40;
      }
      *(_BYTE *)(a3 + 872) = 1;
    }
LABEL_30:
    sub_BA8950((__int64 *)a3);
    goto LABEL_3;
  }
  if ( !v43 )
    goto LABEL_3;
  v26 = *(_QWORD *)(a3 + 32);
  v27 = a3 + 24;
  if ( v26 == a3 + 24 )
  {
    *(_BYTE *)(a3 + 872) = 0;
    goto LABEL_3;
  }
  do
  {
    v28 = v26 - 56;
    if ( !v26 )
      v28 = 0;
    sub_B2B9A0(v28);
    v26 = *(_QWORD *)(v26 + 8);
  }
  while ( v27 != v26 );
  *(_BYTE *)(a3 + 872) = 0;
  v29 = unk_4F81788;
LABEL_40:
  if ( v29 )
    goto LABEL_30;
LABEL_3:
  v44 = sub_BC63A0("*", 1);
  if ( v44 )
  {
    v7 = *(_QWORD *)(a2 + 16);
    if ( v7 )
    {
      v24 = sub_CB6200(*(_QWORD *)a2, *(unsigned __int8 **)(a2 + 8), v7);
      v25 = *(_BYTE **)(v24 + 32);
      if ( *(_BYTE **)(v24 + 24) == v25 )
      {
        sub_CB6200(v24, (unsigned __int8 *)"\n", 1u);
      }
      else
      {
        *v25 = 10;
        ++*(_QWORD *)(v24 + 32);
      }
    }
    sub_A69980((__int64 (__fastcall **)())a3, *(_QWORD *)a2, 0, *(_BYTE *)(a2 + 40), 0, a5);
  }
  else
  {
    for ( i = *(_QWORD *)(a3 + 32); a3 + 24 != i; i = *(_QWORD *)(i + 8) )
    {
      v14 = 0;
      if ( i )
        v14 = i - 56;
      v15 = (char *)sub_BD5D20(v14);
      v17 = sub_BC63A0(v15, v16);
      if ( v17 )
      {
        if ( !v44 )
        {
          v18 = *(_QWORD *)(a2 + 16);
          if ( v18 )
          {
            v45 = v17;
            v19 = sub_CB6200(*(_QWORD *)a2, *(unsigned __int8 **)(a2 + 8), v18);
            v13 = v45;
            v20 = v19;
            v21 = *(_BYTE **)(v19 + 32);
            if ( *(_BYTE **)(v20 + 24) == v21 )
            {
              sub_CB6200(v20, (unsigned __int8 *)"\n", 1u);
              v13 = v45;
            }
            else
            {
              *v21 = 10;
              ++*(_QWORD *)(v20 + 32);
            }
            v44 = v13;
          }
        }
        sub_A68C30(v14, *(_QWORD *)a2, 0, 0, 0);
      }
    }
  }
  if ( *(_BYTE *)(a2 + 41) )
  {
    v22 = sub_BC0510(a4, &unk_4F87818, a3);
    v23 = v22 + 8;
    if ( *(_DWORD *)(v22 + 68) )
    {
LABEL_32:
      sub_A6AF10(v23, *(_QWORD *)a2, 0);
      goto LABEL_7;
    }
    v47 = 0u;
    v48 = 0;
    v35 = sub_C92610();
    v36 = sub_C92740(v22 + 56, byte_3F871B3, 0, v35);
    v23 = v22 + 8;
    v37 = (__int64 *)(*(_QWORD *)(v22 + 56) + 8LL * v36);
    if ( *v37 )
    {
      if ( *v37 != -8 )
        goto LABEL_32;
      --*(_DWORD *)(v22 + 72);
    }
    v42 = v37;
    v46 = v36;
    v38 = sub_C7D670(33, 8);
    v39 = _mm_loadu_si128(&v47);
    v40 = v48;
    *(_BYTE *)(v38 + 32) = 0;
    *(_QWORD *)v38 = 0;
    *(_DWORD *)(v38 + 24) = v40;
    *(__m128i *)(v38 + 8) = v39;
    *v42 = v38;
    ++*(_DWORD *)(v22 + 68);
    sub_C929D0((__int64 *)(v22 + 56), v46);
    v23 = v22 + 8;
    goto LABEL_32;
  }
LABEL_7:
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_QWORD *)(a1 + 16) = 0x100000002LL;
  *(_QWORD *)(a1 + 64) = 2;
  *(_DWORD *)(a1 + 72) = 0;
  *(_BYTE *)(a1 + 76) = 1;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)(a1 + 32) = &qword_4F82400;
  *(_QWORD *)a1 = 1;
  v8 = *(_BYTE *)(a3 + 872);
  if ( v43 )
  {
    if ( !v8 )
    {
      for ( j = *(_QWORD *)(a3 + 32); a3 + 24 != j; j = *(_QWORD *)(j + 8) )
      {
        v11 = j - 56;
        if ( !j )
          v11 = 0;
        sub_B2B950(v11);
      }
      *(_BYTE *)(a3 + 872) = 1;
    }
  }
  else if ( v8 )
  {
    for ( k = *(_QWORD *)(a3 + 32); a3 + 24 != k; k = *(_QWORD *)(k + 8) )
    {
      v31 = k - 56;
      if ( !k )
        v31 = 0;
      sub_B2B9A0(v31);
    }
    *(_BYTE *)(a3 + 872) = 0;
  }
  return a1;
}
