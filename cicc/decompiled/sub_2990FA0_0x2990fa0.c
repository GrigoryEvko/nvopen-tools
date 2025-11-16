// Function: sub_2990FA0
// Address: 0x2990fa0
//
__int64 __fastcall sub_2990FA0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // r13
  unsigned __int64 v4; // r13
  __int64 v6; // r12
  __int64 v7; // rdx
  unsigned int v8; // esi
  __int64 v9; // r15
  int v10; // r11d
  __int64 v11; // r9
  __int64 *v12; // rcx
  unsigned int v13; // eax
  __int64 *v14; // rdi
  __int64 v15; // r8
  __int64 *v16; // r12
  __int64 v17; // rsi
  unsigned __int64 v18; // rax
  __int64 v19; // r15
  int v20; // r13d
  unsigned int v21; // r14d
  unsigned int v22; // esi
  __int64 v23; // rax
  int v24; // eax
  int v25; // edi
  _QWORD *v26; // [rsp+0h] [rbp-50h]
  __int64 v27; // [rsp+8h] [rbp-48h] BYREF
  __int64 *v28; // [rsp+18h] [rbp-38h] BYREF

  result = a2 + 48;
  v3 = *(_QWORD *)(a2 + 48);
  v27 = a2;
  v4 = v3 & 0xFFFFFFFFFFFFFFF8LL;
  if ( v4 == a2 + 48 )
    return result;
  if ( !v4 )
    goto LABEL_24;
  v26 = (_QWORD *)(v4 - 24);
  result = (unsigned int)*(unsigned __int8 *)(v4 - 24) - 30;
  if ( (unsigned int)result > 0xA )
    return result;
  v6 = a2;
  if ( *(_QWORD *)(v4 + 24) )
  {
    v7 = a2;
    v8 = *(_DWORD *)(a1 + 904);
    v9 = a1 + 880;
    if ( v8 )
    {
      v10 = 1;
      v11 = *(_QWORD *)(a1 + 888);
      v12 = 0;
      v13 = (v8 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
      v14 = (__int64 *)(v11 + 16LL * v13);
      v15 = *v14;
      if ( v6 == *v14 )
      {
LABEL_7:
        v16 = v14 + 1;
        goto LABEL_8;
      }
      while ( v15 != -4096 )
      {
        if ( !v12 && v15 == -8192 )
          v12 = v14;
        v13 = (v8 - 1) & (v10 + v13);
        v14 = (__int64 *)(v11 + 16LL * v13);
        v15 = *v14;
        if ( v6 == *v14 )
          goto LABEL_7;
        ++v10;
      }
      v24 = *(_DWORD *)(a1 + 896);
      if ( !v12 )
        v12 = v14;
      ++*(_QWORD *)(a1 + 880);
      v25 = v24 + 1;
      v28 = v12;
      if ( 4 * (v24 + 1) < 3 * v8 )
      {
        if ( v8 - *(_DWORD *)(a1 + 900) - v25 > v8 >> 3 )
        {
LABEL_35:
          *(_DWORD *)(a1 + 896) = v25;
          if ( *v12 != -4096 )
            --*(_DWORD *)(a1 + 900);
          *v12 = v7;
          v16 = v12 + 1;
          v12[1] = 0;
LABEL_8:
          if ( v16 != (__int64 *)(v4 + 24) )
          {
            if ( *v16 )
              sub_B91220((__int64)v16, *v16);
            v17 = *(_QWORD *)(v4 + 24);
            *v16 = v17;
            if ( v17 )
              sub_B96E90((__int64)v16, v17, 1);
          }
          v6 = v27;
          v18 = *(_QWORD *)(v27 + 48) & 0xFFFFFFFFFFFFFFF8LL;
          if ( v18 == v27 + 48 )
            return sub_B43D60(v26);
          if ( v18 )
          {
            v19 = v18 - 24;
            if ( (unsigned int)*(unsigned __int8 *)(v18 - 24) - 30 > 0xA )
              return sub_B43D60(v26);
            goto LABEL_19;
          }
LABEL_24:
          BUG();
        }
LABEL_40:
        sub_298CCE0(v9, v8);
        sub_298BB50(v9, &v27, &v28);
        v7 = v27;
        v12 = v28;
        v25 = *(_DWORD *)(a1 + 896) + 1;
        goto LABEL_35;
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 880);
      v28 = 0;
    }
    v8 *= 2;
    goto LABEL_40;
  }
  v19 = v4 - 24;
LABEL_19:
  v20 = sub_B46E30(v19);
  if ( !v20 )
    return sub_B43D60(v26);
  v21 = 0;
  while ( 1 )
  {
    v22 = v21++;
    v23 = sub_B46EC0(v19, v22);
    sub_2990B70(a1, v6, v23);
    if ( v20 == v21 )
      break;
    v6 = v27;
  }
  return sub_B43D60(v26);
}
