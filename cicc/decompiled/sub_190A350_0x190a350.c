// Function: sub_190A350
// Address: 0x190a350
//
__int64 __fastcall sub_190A350(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 *v3; // r14
  __int64 v4; // r15
  unsigned __int16 *v5; // r8
  unsigned __int16 *v6; // rdi
  unsigned __int16 *v9; // rbx
  unsigned __int16 *v10; // rax
  __int64 v11; // r12
  __int64 v13; // rax
  int v14; // edi
  __int64 v15; // r9
  __int64 v16; // rsi
  __int64 v17; // rdx
  unsigned int v18; // eax
  __int64 *v19; // r10
  __int64 v20; // rdx
  __int64 *v21; // r11
  __int64 v22; // rdi
  unsigned __int16 *v23; // rdx
  unsigned __int16 *v24; // rax
  __int64 v25; // rax
  __int64 v26; // rax
  _QWORD *v27; // rax
  _QWORD *v28; // rbx
  __int64 v29; // rax
  __int64 v30; // rdx
  unsigned __int64 v31; // rax
  __int64 v32; // rax
  unsigned __int16 *v33; // [rsp+8h] [rbp-78h]
  unsigned int v34; // [rsp+1Ch] [rbp-64h] BYREF
  __int64 v35; // [rsp+20h] [rbp-60h] BYREF
  __int64 v36; // [rsp+28h] [rbp-58h] BYREF
  _BYTE v37[16]; // [rsp+30h] [rbp-50h] BYREF
  __int16 v38; // [rsp+40h] [rbp-40h]

  v3 = *(__int64 **)(a1 - 48);
  if ( *((_BYTE *)v3 + 16) == 5 )
    return 0;
  v4 = *v3;
  if ( *(_BYTE *)(*v3 + 8) != 13 )
    return 0;
  v5 = (unsigned __int16 *)*(a2 - 3);
  v6 = *(unsigned __int16 **)(a1 - 24);
  v35 = 0;
  v33 = v5;
  v36 = 0;
  v9 = sub_14AC610(v6, &v35, a3);
  v10 = sub_14AC610(v33, &v36, a3);
  if ( v9 == v10 )
  {
    if ( !v35 )
    {
      if ( v36 )
      {
        v13 = sub_15A9930(a3, v4);
        v34 = 0;
        v14 = *(_DWORD *)(v4 + 12);
        v15 = *a2;
        v16 = v13;
        if ( v14 )
        {
          v17 = 0;
          v18 = 1;
          while ( v36 != *(_QWORD *)(v16 + v17 + 16) || v15 != *(_QWORD *)(*(_QWORD *)(v4 + 16) + v17) )
          {
            v34 = v18;
            v17 += 8;
            if ( v14 == v18 )
              return 0;
            ++v18;
          }
          v38 = 257;
          goto LABEL_34;
        }
      }
    }
    return 0;
  }
  if ( *((_BYTE *)v9 + 16) != 56 || *((_BYTE *)v10 + 16) != 56 )
    return 0;
  v19 = *(__int64 **)&v9[-12 * (*((_DWORD *)v9 + 5) & 0xFFFFFFF)];
  v20 = *v19;
  if ( *(_BYTE *)(*v19 + 8) == 16 )
    v20 = **(_QWORD **)(v20 + 16);
  v21 = *(__int64 **)&v10[-12 * (*((_DWORD *)v10 + 5) & 0xFFFFFFF)];
  v22 = *v21;
  if ( *(_BYTE *)(*v21 + 8) == 16 )
    v22 = **(_QWORD **)(v22 + 16);
  if ( *(_DWORD *)(v22 + 8) >> 8 != *(_DWORD *)(v20 + 8) >> 8
    || v21 != v19
    || (*((_DWORD *)v10 + 5) & 0xFFFFFFF) - 1 != (*((_DWORD *)v9 + 5) & 0xFFFFFFF) )
  {
    return 0;
  }
  v23 = &v10[12 * (1LL - (*((_DWORD *)v10 + 5) & 0xFFFFFFF))];
  v24 = &v9[12 * (1LL - (*((_DWORD *)v9 + 5) & 0xFFFFFFF))];
  if ( v9 != v24 )
  {
    while ( *(_QWORD *)v23 == *(_QWORD *)v24 )
    {
      v24 += 12;
      v23 += 12;
      if ( v9 == v24 )
        goto LABEL_29;
    }
    return 0;
  }
LABEL_29:
  v25 = *(_QWORD *)v23;
  if ( *(_BYTE *)(*(_QWORD *)v23 + 16LL) != 13 )
    return 0;
  v26 = *(_DWORD *)(v25 + 32) <= 0x40u ? *(_QWORD *)(v25 + 24) : **(_QWORD **)(v25 + 24);
  if ( *a2 != *(_QWORD *)(*(_QWORD *)(v4 + 16) + 8LL * (unsigned int)v26) )
    return 0;
  v34 = v26;
  v38 = 257;
LABEL_34:
  v27 = sub_1648A60(88, 1u);
  v11 = (__int64)v27;
  if ( v27 )
  {
    v28 = v27 - 3;
    v29 = sub_15FB2A0(*v3, &v34, 1);
    sub_15F1EA0(v11, v29, 62, v11 - 24, 1, (__int64)a2);
    if ( *(_QWORD *)(v11 - 24) )
    {
      v30 = *(_QWORD *)(v11 - 16);
      v31 = *(_QWORD *)(v11 - 8) & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v31 = v30;
      if ( v30 )
        *(_QWORD *)(v30 + 16) = *(_QWORD *)(v30 + 16) & 3LL | v31;
    }
    *(_QWORD *)(v11 - 24) = v3;
    v32 = v3[1];
    *(_QWORD *)(v11 - 16) = v32;
    if ( v32 )
      *(_QWORD *)(v32 + 16) = (v11 - 16) | *(_QWORD *)(v32 + 16) & 3LL;
    *(_QWORD *)(v11 - 8) = (unsigned __int64)(v3 + 1) | *(_QWORD *)(v11 - 8) & 3LL;
    v3[1] = (__int64)v28;
    *(_QWORD *)(v11 + 56) = v11 + 72;
    *(_QWORD *)(v11 + 64) = 0x400000000LL;
    sub_15FB110(v11, &v34, 1, (__int64)v37);
  }
  return v11;
}
