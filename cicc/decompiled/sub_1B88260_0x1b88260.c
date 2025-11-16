// Function: sub_1B88260
// Address: 0x1b88260
//
char __fastcall sub_1B88260(__int64 a1, __int64 a2, __int64 a3, __m128i a4, __m128i a5)
{
  __int64 *v6; // r13
  unsigned __int8 v7; // al
  unsigned __int8 v8; // al
  char result; // al
  __int64 v10; // r14
  unsigned int v11; // r15d
  __int64 v12; // r9
  int v13; // eax
  __int64 *v14; // rax
  unsigned __int64 v15; // rbx
  __int64 v16; // r9
  __int64 v17; // rsi
  __int64 v18; // rax
  __int64 v19; // r8
  unsigned int v20; // ebx
  __int64 v21; // rax
  unsigned __int64 v22; // rsi
  __int64 v23; // rax
  int v24; // eax
  __int64 v25; // rax
  int v26; // eax
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // [rsp+0h] [rbp-60h]
  __int64 v30; // [rsp+0h] [rbp-60h]
  __int64 v31; // [rsp+8h] [rbp-58h]
  __int64 v32; // [rsp+8h] [rbp-58h]
  char v33; // [rsp+8h] [rbp-58h]
  char v34; // [rsp+8h] [rbp-58h]
  unsigned __int64 v35; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v36; // [rsp+18h] [rbp-48h]
  unsigned __int64 v37; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v38; // [rsp+28h] [rbp-38h]

  v6 = 0;
  v7 = *(_BYTE *)(a2 + 16);
  if ( v7 > 0x17u )
  {
    if ( (unsigned __int8)(v7 - 54) <= 1u )
    {
      v6 = *(__int64 **)(a2 - 24);
    }
    else if ( v7 == 78 )
    {
      v23 = *(_QWORD *)(a2 - 24);
      if ( !*(_BYTE *)(v23 + 16) )
      {
        v24 = *(_DWORD *)(v23 + 36);
        if ( v24 == 4085 || v24 == 4057 )
        {
          v6 = *(__int64 **)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
        }
        else if ( v24 == 4503 || v24 == 4492 )
        {
          v6 = *(__int64 **)(a2 + 24 * (2LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
        }
      }
    }
  }
  v8 = *(_BYTE *)(a3 + 16);
  if ( v8 <= 0x17u )
    return 0;
  if ( v8 == 54 || v8 == 55 )
  {
    v10 = *(_QWORD *)(a3 - 24);
    if ( !v10 )
      return 0;
  }
  else
  {
    if ( v8 != 78 )
      return 0;
    v25 = *(_QWORD *)(a3 - 24);
    if ( *(_BYTE *)(v25 + 16) )
      return 0;
    v26 = *(_DWORD *)(v25 + 36);
    if ( v26 == 4085 || v26 == 4057 )
    {
      v27 = 1;
      v28 = *(_DWORD *)(a3 + 20) & 0xFFFFFFF;
    }
    else
    {
      if ( v26 != 4503 && v26 != 4492 )
        return 0;
      v27 = 2;
      v28 = *(_DWORD *)(a3 + 20) & 0xFFFFFFF;
    }
    v10 = *(_QWORD *)(a3 + 24 * (v27 - v28));
    if ( !v10 )
      return 0;
  }
  v11 = sub_1B7C940(a2);
  v13 = sub_1B7C940(v12);
  if ( !v6 )
    return 0;
  if ( v11 != v13 )
    return 0;
  if ( (__int64 *)v10 == v6 )
    return 0;
  v14 = *(__int64 **)(*(_QWORD *)v10 + 16LL);
  if ( (*(_BYTE *)(**(_QWORD **)(*v6 + 16) + 8LL) == 16) != (*(_BYTE *)(*v14 + 8) == 16) )
    return 0;
  v31 = *v14;
  v29 = **(_QWORD **)(*v6 + 16);
  v15 = sub_127FA20(*(_QWORD *)(a1 + 40), v29) + 7;
  if ( (unsigned __int64)(sub_127FA20(*(_QWORD *)(a1 + 40), v31) + 7) >> 3 != v15 >> 3 )
    return 0;
  v16 = v29;
  v17 = v29;
  if ( *(_BYTE *)(v29 + 8) == 16 )
    v17 = **(_QWORD **)(v29 + 16);
  v30 = v31;
  v32 = v16;
  v18 = sub_127FA20(*(_QWORD *)(a1 + 40), v17);
  v19 = v30;
  if ( *(_BYTE *)(v30 + 8) == 16 )
    v19 = **(_QWORD **)(v30 + 16);
  if ( (unsigned __int64)(sub_127FA20(*(_QWORD *)(a1 + 40), v19) + 7) >> 3 != (unsigned __int64)(v18 + 7) >> 3 )
    return 0;
  v20 = 8 * sub_15A9520(*(_QWORD *)(a1 + 40), v11);
  v21 = sub_127FA20(*(_QWORD *)(a1 + 40), v32);
  v36 = v20;
  v22 = (unsigned __int64)(v21 + 7) >> 3;
  if ( v20 <= 0x40 )
  {
    v38 = v20;
    v35 = v22 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v20);
LABEL_23:
    v37 = v35;
    goto LABEL_24;
  }
  sub_16A4EF0((__int64)&v35, v22, 0);
  v38 = v36;
  if ( v36 <= 0x40 )
    goto LABEL_23;
  sub_16A4FD0((__int64)&v37, (const void **)&v35);
LABEL_24:
  result = sub_1B86990(a1, v6, v10, (__int64 *)&v37, 0, a4, a5);
  if ( v38 > 0x40 && v37 )
  {
    v33 = result;
    j_j___libc_free_0_0(v37);
    result = v33;
  }
  if ( v36 > 0x40 )
  {
    if ( v35 )
    {
      v34 = result;
      j_j___libc_free_0_0(v35);
      return v34;
    }
  }
  return result;
}
