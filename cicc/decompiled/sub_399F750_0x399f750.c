// Function: sub_399F750
// Address: 0x399f750
//
__int64 __fastcall sub_399F750(__int64 a1, __int64 a2, unsigned __int64 **a3, unsigned int a4)
{
  unsigned int v6; // ecx
  unsigned int v7; // r13d
  unsigned __int64 *v8; // rdx
  int *v9; // r12
  int *v10; // r14
  unsigned int v11; // esi
  char v13; // r14
  int v14; // r15d
  unsigned __int64 v15; // rax
  unsigned __int64 *v16; // rdx
  unsigned __int64 *v17; // rax
  unsigned __int64 v18; // rax
  int v19; // eax
  unsigned __int64 *v20; // [rsp+8h] [rbp-78h]
  unsigned __int64 *v21; // [rsp+10h] [rbp-70h]
  char v22; // [rsp+10h] [rbp-70h]
  unsigned __int64 *v23; // [rsp+10h] [rbp-70h]
  unsigned __int64 *v24; // [rsp+10h] [rbp-70h]
  int v25; // [rsp+18h] [rbp-68h]
  unsigned __int64 *v26; // [rsp+28h] [rbp-58h] BYREF
  unsigned int v27[4]; // [rsp+30h] [rbp-50h] BYREF
  char v28; // [rsp+40h] [rbp-40h]

  sub_15B1350((__int64)v27, *a3, a3[1]);
  v6 = -2;
  if ( v28 )
    v6 = v27[0];
  v7 = sub_399EB70(a1, a2, a4, v6);
  if ( !(_BYTE)v7 )
  {
    *(_DWORD *)(a1 + 76) = 0;
    return v7;
  }
  v20 = *a3;
  v21 = a3[1];
  if ( v21 == *a3 )
  {
    if ( *(_DWORD *)(a1 + 76) == 2 )
    {
      if ( *(_DWORD *)(a1 + 64) <= 3u )
      {
        v22 = 0;
LABEL_30:
        v25 = **(_DWORD **)(a1 + 8);
        v13 = (*(__int64 (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)a1 + 32LL))(a1, a2, a4);
        if ( v22 )
          goto LABEL_31;
      }
      else
      {
        v25 = **(_DWORD **)(a1 + 8);
        v13 = (*(__int64 (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)a1 + 32LL))(a1, a2, a4);
      }
      v14 = 0;
      goto LABEL_26;
    }
    goto LABEL_13;
  }
  if ( *v20 == 4096 )
  {
    if ( *(_DWORD *)(a1 + 76) == 2 )
      goto LABEL_7;
LABEL_13:
    v9 = *(int **)(a1 + 8);
    v10 = &v9[4 * *(unsigned int *)(a1 + 16)];
    while ( v10 != v9 )
    {
      if ( *v9 >= 0 || *(_BYTE *)(a1 + 80) )
        sub_399E900(a1, *v9);
      v11 = v9[1];
      v9 += 4;
      sub_399EA60(a1, v11, 0);
    }
    goto LABEL_18;
  }
  if ( *(_DWORD *)(a1 + 16) <= 1u )
  {
LABEL_7:
    if ( *(_DWORD *)(a1 + 64) <= 3u )
    {
      v26 = *a3;
      v8 = v20;
      while ( *v8 != 159 )
      {
        v8 += (unsigned int)sub_15B11B0(&v26);
        v26 = v8;
        if ( v21 == v8 )
        {
          v22 = v7;
          goto LABEL_30;
        }
      }
      goto LABEL_11;
    }
    v25 = **(_DWORD **)(a1 + 8);
    v13 = (*(__int64 (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)a1 + 32LL))(a1, a2, a4);
LABEL_31:
    v14 = 0;
    v15 = *v20;
    if ( *v20 == 35 )
    {
      v14 = *((_DWORD *)v20 + 2);
      if ( *a3 == a3[1] )
        goto LABEL_26;
      v24 = *a3;
      *a3 = &v24[(unsigned int)sub_15B11B0(a3)];
      v15 = *v20;
    }
    if ( v15 != 16 )
      goto LABEL_26;
    v16 = *a3;
    if ( *a3 == a3[1] )
      goto LABEL_26;
    v26 = *a3;
    v17 = &v16[(unsigned int)sub_15B11B0(&v26)];
    if ( a3[1] == v17 )
      goto LABEL_26;
    v18 = *v17;
    if ( v18 == 34 )
    {
      v14 = *((_DWORD *)v20 + 2);
    }
    else
    {
      if ( v18 != 28 || *(_DWORD *)(a1 + 68) )
        goto LABEL_26;
      v14 = -*((_DWORD *)v20 + 2);
    }
    v23 = *a3;
    v19 = sub_15B11B0(a3);
    *a3 = &v23[v19];
    *a3 = &v23[v19 + (unsigned int)sub_15B11B0(a3)];
LABEL_26:
    if ( v13 )
      sub_399EA30(a1, v14);
    else
      sub_399E990(a1, v25, v14);
LABEL_18:
    *(_DWORD *)(a1 + 16) = 0;
    return v7;
  }
LABEL_11:
  *(_DWORD *)(a1 + 16) = 0;
  v7 = 0;
  *(_DWORD *)(a1 + 76) = 0;
  return v7;
}
