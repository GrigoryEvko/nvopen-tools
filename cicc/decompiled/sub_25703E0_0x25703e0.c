// Function: sub_25703E0
// Address: 0x25703e0
//
__int64 __fastcall sub_25703E0(__int64 a1, char *a2)
{
  _QWORD *v4; // rdi
  __int64 v6; // rsi
  __int64 result; // rax
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  _QWORD *v12; // r12
  _QWORD *v13; // r14
  unsigned int v14; // esi
  int v15; // eax
  unsigned __int64 *v16; // r8
  int v17; // eax
  unsigned int v18; // esi
  int v19; // eax
  unsigned __int64 *v20; // r13
  int v21; // eax
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // r9
  unsigned __int64 *v26; // [rsp+8h] [rbp-68h]
  unsigned __int64 *v27; // [rsp+10h] [rbp-60h] BYREF
  unsigned __int64 *v28; // [rsp+18h] [rbp-58h] BYREF
  __int64 v29; // [rsp+20h] [rbp-50h] BYREF
  __int64 v30; // [rsp+28h] [rbp-48h]
  __int64 v31; // [rsp+30h] [rbp-40h]

  if ( *(_DWORD *)(a1 + 16) )
  {
    result = sub_25116B0(a1, (__int64)a2, &v27);
    if ( (_BYTE)result )
      return result;
    v18 = *(_DWORD *)(a1 + 24);
    v19 = *(_DWORD *)(a1 + 16);
    v20 = v27;
    ++*(_QWORD *)a1;
    v21 = v19 + 1;
    v28 = v20;
    if ( 4 * v21 >= 3 * v18 )
    {
      v18 *= 2;
    }
    else if ( v18 - *(_DWORD *)(a1 + 20) - v21 > v18 >> 3 )
    {
LABEL_16:
      *(_DWORD *)(a1 + 16) = v21;
      v29 = 4;
      v30 = 0;
      v31 = -4096;
      if ( v20[2] != -4096 )
        --*(_DWORD *)(a1 + 20);
      sub_D68D70(&v29);
      sub_2538AB0(v20, a2);
      return sub_2568C30(a1 + 32, a2, v22, v23, v24, v25);
    }
    sub_2517BE0(a1, v18);
    sub_25116B0(a1, (__int64)a2, &v28);
    v20 = v28;
    v21 = *(_DWORD *)(a1 + 16) + 1;
    goto LABEL_16;
  }
  v4 = *(_QWORD **)(a1 + 32);
  v6 = (__int64)&v4[3 * *(unsigned int *)(a1 + 40)];
  result = (__int64)sub_2538140(v4, v6, (__int64)a2);
  if ( v6 == result )
  {
    sub_2568C30(a1 + 32, a2, v8, v9, v10, v11);
    result = *(unsigned int *)(a1 + 40);
    if ( (unsigned int)result > 0x10 )
    {
      v12 = *(_QWORD **)(a1 + 32);
      v13 = &v12[3 * result];
      while ( 1 )
      {
        result = sub_25116B0(a1, (__int64)v12, &v27);
        if ( !(_BYTE)result )
          break;
LABEL_6:
        v12 += 3;
        if ( v13 == v12 )
          return result;
      }
      v14 = *(_DWORD *)(a1 + 24);
      v15 = *(_DWORD *)(a1 + 16);
      v16 = v27;
      ++*(_QWORD *)a1;
      v17 = v15 + 1;
      v28 = v16;
      if ( 4 * v17 >= 3 * v14 )
      {
        v14 *= 2;
      }
      else if ( v14 - *(_DWORD *)(a1 + 20) - v17 > v14 >> 3 )
      {
LABEL_10:
        *(_DWORD *)(a1 + 16) = v17;
        v29 = 4;
        v30 = 0;
        v31 = -4096;
        if ( v16[2] != -4096 )
          --*(_DWORD *)(a1 + 20);
        v26 = v16;
        sub_D68D70(&v29);
        result = sub_2538AB0(v26, v12);
        goto LABEL_6;
      }
      sub_2517BE0(a1, v14);
      sub_25116B0(a1, (__int64)v12, &v28);
      v16 = v28;
      v17 = *(_DWORD *)(a1 + 16) + 1;
      goto LABEL_10;
    }
  }
  return result;
}
