// Function: sub_1E11B70
// Address: 0x1e11b70
//
__int64 __fastcall sub_1E11B70(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 v6; // r12
  unsigned int v7; // eax
  int v8; // r8d
  int v9; // r9d
  unsigned int v10; // r14d
  unsigned int v11; // r13d
  unsigned int v12; // eax
  __int64 v13; // rax
  unsigned int v14; // r14d
  unsigned int v15; // edx
  __int64 v16; // rax
  __int64 v17; // rsi
  __int64 v18; // rdx
  _QWORD *v19; // rcx
  unsigned __int64 v21; // r15
  unsigned int v22; // r8d
  unsigned int v23; // edx
  unsigned __int64 v24; // rdx
  __int64 v25; // rax
  unsigned __int64 v26; // rdx
  int v27; // ecx
  unsigned int v28; // r8d
  int v29; // ecx
  __int64 v30; // rdx
  unsigned __int64 v31; // rax
  unsigned int v32; // r15d
  int v33; // r14d
  unsigned __int64 v34; // rdx
  unsigned __int64 v35; // rax
  unsigned __int64 v36; // r15
  unsigned __int64 v37; // [rsp+0h] [rbp-40h]
  __int64 v38; // [rsp+8h] [rbp-38h]
  unsigned int v39; // [rsp+8h] [rbp-38h]
  unsigned int v40; // [rsp+8h] [rbp-38h]

  if ( (*(_BYTE *)(a2 + 32) & 0xF) == 1 )
    return 0;
  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_44:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_4FC6A0E )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_44;
  }
  v5 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_4FC6A0E);
  v6 = sub_1E305C0(v5, a2);
  v7 = (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 152LL))(a1, v6);
  v10 = *(_DWORD *)(a1 + 200);
  v11 = v7;
  v12 = *(_DWORD *)(v6 + 368);
  if ( v12 < v10 )
  {
    v21 = *(_QWORD *)(v6 + 360);
    if ( v10 > v21 << 6 )
    {
      v24 = (v10 + 63) >> 6;
      if ( v24 < 2 * v21 )
        v24 = 2 * v21;
      v37 = v24;
      v38 = 8 * v24;
      v25 = (__int64)realloc(*(_QWORD *)(v6 + 352), 8 * v24, v24, 8 * (int)v24, v8, v9);
      v26 = v37;
      if ( !v25 )
      {
        if ( v38 )
        {
          sub_16BD1C0("Allocation failed", 1u);
          v25 = 0;
          v26 = v37;
        }
        else
        {
          v25 = malloc(1u);
          v26 = v37;
          if ( !v25 )
          {
            sub_16BD1C0("Allocation failed", 1u);
            v26 = v37;
            v25 = 0;
          }
        }
      }
      v27 = *(_DWORD *)(v6 + 368);
      *(_QWORD *)(v6 + 352) = v25;
      *(_QWORD *)(v6 + 360) = v26;
      v28 = (unsigned int)(v27 + 63) >> 6;
      if ( v26 > v28 )
      {
        v34 = v26 - v28;
        if ( v34 )
        {
          v39 = (unsigned int)(v27 + 63) >> 6;
          memset((void *)(v25 + 8LL * v28), 0, 8 * v34);
          v27 = *(_DWORD *)(v6 + 368);
          v28 = v39;
          v25 = *(_QWORD *)(v6 + 352);
        }
      }
      v29 = v27 & 0x3F;
      if ( v29 )
      {
        *(_QWORD *)(v25 + 8LL * (v28 - 1)) &= ~(-1LL << v29);
        v25 = *(_QWORD *)(v6 + 352);
      }
      v30 = *(_QWORD *)(v6 + 360) - (unsigned int)v21;
      if ( v30 )
        memset((void *)(v25 + 8LL * (unsigned int)v21), 0, 8 * v30);
      v12 = *(_DWORD *)(v6 + 368);
      v23 = v12;
      if ( v10 <= v12 )
        goto LABEL_19;
      v21 = *(_QWORD *)(v6 + 360);
    }
    v22 = (v12 + 63) >> 6;
    if ( v21 > v22 )
    {
      v36 = v21 - v22;
      if ( v36 )
      {
        v40 = (v12 + 63) >> 6;
        memset((void *)(*(_QWORD *)(v6 + 352) + 8LL * v22), 0, 8 * v36);
        v12 = *(_DWORD *)(v6 + 368);
        v22 = v40;
      }
    }
    v23 = v12;
    if ( (v12 & 0x3F) != 0 )
    {
      *(_QWORD *)(*(_QWORD *)(v6 + 352) + 8LL * (v22 - 1)) &= ~(-1LL << (v12 & 0x3F));
      v23 = *(_DWORD *)(v6 + 368);
    }
LABEL_19:
    *(_DWORD *)(v6 + 368) = v10;
    if ( v23 > v10 )
    {
      v31 = *(_QWORD *)(v6 + 360);
      v32 = (v10 + 63) >> 6;
      if ( v31 > v32 )
      {
        v35 = v31 - v32;
        if ( v35 )
        {
          memset((void *)(*(_QWORD *)(v6 + 352) + 8LL * v32), 0, 8 * v35);
          v10 = *(_DWORD *)(v6 + 368);
        }
      }
      v33 = v10 & 0x3F;
      if ( v33 )
        *(_QWORD *)(*(_QWORD *)(v6 + 352) + 8LL * (v32 - 1)) &= ~(-1LL << v33);
    }
    v10 = *(_DWORD *)(a1 + 200);
  }
  v13 = 0;
  v14 = (v10 + 63) >> 6;
  if ( v14 )
  {
    do
    {
      *(_QWORD *)(*(_QWORD *)(v6 + 352) + 8 * v13) |= *(_QWORD *)(*(_QWORD *)(a1 + 184) + 8 * v13);
      ++v13;
    }
    while ( v14 != v13 );
  }
  v15 = (unsigned int)(*(_DWORD *)(a1 + 224) + 63) >> 6;
  if ( (unsigned int)(*(_DWORD *)(v6 + 368) + 63) >> 6 <= v15 )
    v15 = (unsigned int)(*(_DWORD *)(v6 + 368) + 63) >> 6;
  v16 = 0;
  v17 = 8LL * v15;
  if ( v15 )
  {
    do
    {
      v18 = *(_QWORD *)(*(_QWORD *)(a1 + 208) + v16);
      v19 = (_QWORD *)(v16 + *(_QWORD *)(v6 + 352));
      v16 += 8;
      *v19 &= ~v18;
    }
    while ( v17 != v16 );
  }
  return v11;
}
