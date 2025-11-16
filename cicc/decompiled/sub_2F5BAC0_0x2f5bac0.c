// Function: sub_2F5BAC0
// Address: 0x2f5bac0
//
__int64 __fastcall sub_2F5BAC0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  int v7; // r15d
  _QWORD *v8; // rdi
  unsigned int v9; // r10d
  int v10; // r11d
  __int64 v11; // r13
  __int64 v12; // r9
  unsigned __int16 *v13; // rsi
  __int64 v14; // r9
  int v15; // r11d
  unsigned __int16 **v16; // r15
  unsigned int v17; // r12d
  __int64 result; // rax
  __int64 v19; // rdx
  __int64 v20; // rax
  unsigned __int8 v21; // r8
  __int64 v22; // rax
  unsigned int *v23; // rdx
  unsigned int v24; // ebx
  unsigned __int16 *v25; // rdi
  unsigned __int16 *v26; // rax
  __int64 v27; // r8
  int v30; // [rsp+14h] [rbp-4Ch]
  __int64 v32[7]; // [rsp+28h] [rbp-38h] BYREF

  v30 = *(_DWORD *)(a3 + 72);
  if ( -*(_DWORD *)(a3 + 8) == v30 )
    return 0;
  v7 = -*(_DWORD *)(a3 + 8);
  while ( 1 )
  {
    v8 = *(_QWORD **)(a1 + 40);
    if ( v7 < 0 )
    {
      if ( !(unsigned int)sub_2E21680(v8, a2, *(unsigned __int16 *)(*(_QWORD *)a3 + 2 * (v7 + *(_QWORD *)(a3 + 8)))) )
        return *(unsigned __int16 *)(*(_QWORD *)a3 + 2 * (v7 + *(_QWORD *)(a3 + 8)));
LABEL_19:
      v9 = 0;
      goto LABEL_8;
    }
    if ( (unsigned int)sub_2E21680(v8, a2, *(unsigned __int16 *)(*(_QWORD *)(a3 + 56) + 2LL * v7)) )
      goto LABEL_19;
    v9 = *(unsigned __int16 *)(*(_QWORD *)(a3 + 56) + 2LL * v7);
LABEL_8:
    v10 = *(_DWORD *)(a3 + 72);
    v7 += v10 > v7;
    if ( v7 < v10 && v7 >= 0 )
    {
      v11 = *(_QWORD *)(a3 + 56);
      v12 = v7;
      while ( 1 )
      {
        v7 = v12;
        if ( (unsigned int)*(unsigned __int16 *)(v11 + 2 * v12) - 1 > 0x3FFFFFFE )
          break;
        LODWORD(v32[0]) = *(unsigned __int16 *)(v11 + 2 * v12);
        v13 = (unsigned __int16 *)(*(_QWORD *)a3 + 2LL * *(_QWORD *)(a3 + 8));
        if ( v13 == sub_2F4C810(*(unsigned __int16 **)a3, (__int64)v13, (int *)v32) )
          break;
        v12 = v14 + 1;
        ++v7;
        if ( v15 <= (int)v12 )
        {
          if ( v30 != v7 )
            goto LABEL_4;
          goto LABEL_15;
        }
      }
    }
    if ( v30 == v7 )
      break;
LABEL_4:
    if ( v9 )
    {
      v16 = (unsigned __int16 **)a3;
      v17 = v9;
      goto LABEL_21;
    }
  }
LABEL_15:
  v16 = (unsigned __int16 **)a3;
  v17 = v9;
  if ( !v9 )
    return 0;
LABEL_21:
  v19 = *(_QWORD *)(a1 + 16);
  v20 = *(_DWORD *)(a2 + 112) & 0x7FFFFFFF;
  if ( (unsigned int)v20 >= *(_DWORD *)(v19 + 248)
    || (v22 = *(_QWORD *)(v19 + 240) + 40 * v20, !*(_DWORD *)(v22 + 16))
    || (v23 = *(unsigned int **)(v22 + 8), v24 = *v23, *(_DWORD *)v22)
    || !v24
    || v24 - 1 > 0x3FFFFFFE
    || (v25 = *v16,
        v26 = v16[1],
        LODWORD(v32[0]) = *v23,
        &v25[(_QWORD)v26] == sub_2F4C810(v25, (__int64)&v25[(_QWORD)v26], (int *)v32)) )
  {
LABEL_22:
    v21 = *(_BYTE *)(*(_QWORD *)(a1 + 29064) + v17);
    result = v17;
    if ( v21 )
    {
      result = sub_2F51120(a1, a2, (__int64)v16, a4, v21, a5);
      if ( !(_DWORD)result )
        return v17;
    }
  }
  else
  {
    if ( (*(unsigned __int8 (__fastcall **)(_QWORD, __int64, _QWORD, __int64))(**(_QWORD **)(a1 + 968) + 24LL))(
           *(_QWORD *)(a1 + 968),
           a2,
           v24,
           a5) )
    {
      sub_2F50AD0(a1, a2, v24, a4, v27);
      return v24;
    }
    if ( !(unsigned __int8)sub_2F580B0(a1, v24, a2, a4, (__int64)v16) )
    {
      v32[0] = a2;
      sub_2F5B790(a1 + 28952, v32);
      goto LABEL_22;
    }
    return 0;
  }
  return result;
}
