// Function: sub_E7EF30
// Address: 0xe7ef30
//
__int64 __fastcall sub_E7EF30(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r13
  __int64 v4; // r12
  _QWORD *v5; // rdx
  __int64 v6; // rax
  __int64 v7; // rbx
  __int64 v8; // rdx
  __int64 v9; // r15
  __int64 v10; // r12
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r15
  __int64 v16; // r13
  unsigned __int8 *v17; // rsi
  __int64 result; // rax
  char v19; // dl
  __int64 v20; // rax
  __int64 v21; // [rsp+8h] [rbp-48h]
  __int64 v22; // [rsp+10h] [rbp-40h]

  v3 = a1[37];
  if ( *(_DWORD *)(v3 + 368) )
  {
    v4 = *(_QWORD *)(a1[36] + 8LL);
    if ( sub_E7E4B0((__int64)a1) && (*(_BYTE *)(v4 + 48) & 1) == 0 )
    {
      v7 = a1[36];
      v20 = *(_QWORD *)(v7 + 32);
      if ( v20 && a3 != v20 )
        sub_C64ED0("A Bundle can only have one Subtarget.", 1u);
    }
    else
    {
      v5 = (_QWORD *)a1[1];
      v6 = v5[36];
      v5[46] += 208LL;
      v7 = (v6 + 7) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v5[37] >= (unsigned __int64)(v7 + 208) && v6 )
        v5[36] = v7 + 208;
      else
        v7 = sub_9D1E70((__int64)(v5 + 36), 208, 208, 3);
      sub_E81B30(v7, 1, 0);
      *(_BYTE *)(v7 + 30) = 0;
      *(_QWORD *)(v7 + 40) = v7 + 64;
      *(_QWORD *)(v7 + 96) = v7 + 112;
      *(_QWORD *)(v7 + 32) = 0;
      *(_QWORD *)(v7 + 48) = 0;
      *(_QWORD *)(v7 + 56) = 32;
      *(_QWORD *)(v7 + 104) = 0x400000000LL;
      v8 = *(_QWORD *)(a1[36] + 8LL);
      *(_QWORD *)(v7 + 8) = v8;
      *(_DWORD *)(v7 + 24) = *(_DWORD *)(a1[36] + 24LL) + 1;
      *(_QWORD *)a1[36] = v7;
      a1[36] = v7;
      *(_QWORD *)(*(_QWORD *)(v8 + 8) + 8LL) = v7;
    }
    if ( *(_DWORD *)(v4 + 40) == 2 )
      *(_BYTE *)(v7 + 29) |= 2u;
    *(_BYTE *)(v4 + 48) &= ~1u;
  }
  else
  {
    v7 = sub_E8BB10(a1, a3);
  }
  v9 = *(unsigned int *)(v7 + 104);
  v10 = *(_QWORD *)(v7 + 48);
  (*(void (__fastcall **)(_QWORD, __int64, __int64, __int64, __int64))(**(_QWORD **)(v3 + 16) + 24LL))(
    *(_QWORD *)(v3 + 16),
    a2,
    v7 + 40,
    v7 + 96,
    a3);
  v13 = 3 * v9;
  v22 = *(unsigned int *)(v7 + 104) - v9;
  v21 = *(_QWORD *)(v7 + 96) + 24 * v9;
  v14 = v21;
  v15 = v21;
  v16 = 24 * v22 + v21;
  if ( v21 != v16 )
  {
    do
    {
      *(_DWORD *)(v15 + 8) += v10;
      v17 = *(unsigned __int8 **)v15;
      v15 += 24;
      sub_E7E6A0((__int64)a1, v17, v13, v14, v11, v12);
    }
    while ( v16 != v15 );
  }
  result = *(unsigned __int8 *)(v7 + 29);
  v19 = *(_BYTE *)(v7 + 29);
  *(_QWORD *)(v7 + 32) = a3;
  *(_BYTE *)(v7 + 29) = v19 | 1;
  if ( v22 )
  {
    if ( *(_DWORD *)(*(_QWORD *)(a1[37] + 8LL) + 12LL) == *(_DWORD *)(v21 + 24 * v22 - 12) )
    {
      result = (unsigned int)result | 5;
      *(_BYTE *)(v7 + 29) = result;
    }
  }
  return result;
}
