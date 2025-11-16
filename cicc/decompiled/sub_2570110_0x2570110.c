// Function: sub_2570110
// Address: 0x2570110
//
__int64 __fastcall sub_2570110(__int64 a1, __int64 a2)
{
  _QWORD *v3; // rdi
  _QWORD *v4; // rsi
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v10; // rax
  _QWORD *v11; // r14
  unsigned int v12; // esi
  int v13; // eax
  unsigned __int64 *v14; // r9
  int v15; // eax
  unsigned int v16; // esi
  int v17; // eax
  unsigned __int64 *v18; // r13
  int v19; // eax
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  unsigned __int64 *v24; // [rsp+10h] [rbp-90h]
  _QWORD *v25; // [rsp+18h] [rbp-88h]
  unsigned __int64 *v26; // [rsp+20h] [rbp-80h] BYREF
  unsigned __int64 *v27; // [rsp+28h] [rbp-78h] BYREF
  _QWORD v28[4]; // [rsp+30h] [rbp-70h] BYREF
  __int64 v29; // [rsp+50h] [rbp-50h] BYREF
  __int64 v30; // [rsp+58h] [rbp-48h]
  __int64 v31; // [rsp+60h] [rbp-40h]

  v28[0] = 4;
  v28[1] = 0;
  v28[2] = a2;
  if ( a2 != -8192 && a2 != -4096 )
    sub_BD73F0((__int64)v28);
  if ( *(_DWORD *)(a1 + 3896) )
  {
    if ( (unsigned __int8)sub_25116B0(a1 + 3880, (__int64)v28, &v26) )
      return sub_D68D70(v28);
    v16 = *(_DWORD *)(a1 + 3904);
    v17 = *(_DWORD *)(a1 + 3896);
    v18 = v26;
    ++*(_QWORD *)(a1 + 3880);
    v19 = v17 + 1;
    v27 = v18;
    if ( 4 * v19 >= 3 * v16 )
    {
      v16 *= 2;
    }
    else if ( v16 - *(_DWORD *)(a1 + 3900) - v19 > v16 >> 3 )
    {
LABEL_19:
      *(_DWORD *)(a1 + 3896) = v19;
      v29 = 4;
      v30 = 0;
      v31 = -4096;
      if ( v18[2] != -4096 )
        --*(_DWORD *)(a1 + 3900);
      sub_D68D70(&v29);
      sub_2538AB0(v18, v28);
      sub_2568C30(a1 + 3912, (char *)v28, v20, v21, v22, v23);
      return sub_D68D70(v28);
    }
    sub_2517BE0(a1 + 3880, v16);
    sub_25116B0(a1 + 3880, (__int64)v28, &v27);
    v18 = v27;
    v19 = *(_DWORD *)(a1 + 3896) + 1;
    goto LABEL_19;
  }
  v3 = *(_QWORD **)(a1 + 3912);
  v4 = &v3[3 * *(unsigned int *)(a1 + 3920)];
  if ( v4 == sub_2538140(v3, (__int64)v4, (__int64)v28) )
  {
    sub_2568C30(a1 + 3912, (char *)v28, v5, v6, v7, v8);
    v10 = *(unsigned int *)(a1 + 3920);
    if ( (unsigned int)v10 > 8 )
    {
      v11 = *(_QWORD **)(a1 + 3912);
      v25 = &v11[3 * v10];
      while ( (unsigned __int8)sub_25116B0(a1 + 3880, (__int64)v11, &v26) )
      {
LABEL_9:
        v11 += 3;
        if ( v25 == v11 )
          return sub_D68D70(v28);
      }
      v12 = *(_DWORD *)(a1 + 3904);
      v13 = *(_DWORD *)(a1 + 3896);
      v14 = v26;
      ++*(_QWORD *)(a1 + 3880);
      v15 = v13 + 1;
      v27 = v14;
      if ( 4 * v15 >= 3 * v12 )
      {
        v12 *= 2;
      }
      else if ( v12 - *(_DWORD *)(a1 + 3900) - v15 > v12 >> 3 )
      {
LABEL_13:
        *(_DWORD *)(a1 + 3896) = v15;
        v29 = 4;
        v30 = 0;
        v31 = -4096;
        if ( v14[2] != -4096 )
          --*(_DWORD *)(a1 + 3900);
        v24 = v14;
        sub_D68D70(&v29);
        sub_2538AB0(v24, v11);
        goto LABEL_9;
      }
      sub_2517BE0(a1 + 3880, v12);
      sub_25116B0(a1 + 3880, (__int64)v11, &v27);
      v14 = v27;
      v15 = *(_DWORD *)(a1 + 3896) + 1;
      goto LABEL_13;
    }
  }
  return sub_D68D70(v28);
}
