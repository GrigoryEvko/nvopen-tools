// Function: sub_19EE580
// Address: 0x19ee580
//
_QWORD *__fastcall sub_19EE580(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r15
  unsigned int v5; // esi
  __int64 v6; // rdx
  __int64 v7; // rcx
  unsigned int v8; // eax
  __int64 *v9; // rbx
  __int64 v10; // r8
  _QWORD *v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rcx
  int v15; // r10d
  __int64 *v16; // r9
  int v17; // eax
  int v18; // ecx
  __int64 v19; // [rsp+8h] [rbp-58h] BYREF
  _QWORD v20[10]; // [rsp+10h] [rbp-50h] BYREF

  v3 = a1 + 1928;
  v19 = a2;
  v5 = *(_DWORD *)(a1 + 1952);
  if ( !v5 )
  {
    ++*(_QWORD *)(a1 + 1928);
LABEL_17:
    sub_19EE3D0(v3, 2 * v5);
LABEL_18:
    sub_19EB070(v3, &v19, v20);
    v9 = (__int64 *)v20[0];
    v6 = v19;
    v18 = *(_DWORD *)(a1 + 1944) + 1;
    goto LABEL_13;
  }
  v6 = v19;
  v7 = *(_QWORD *)(a1 + 1936);
  v8 = (v5 - 1) & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
  v9 = (__int64 *)(v7 + ((unsigned __int64)v8 << 6));
  v10 = *v9;
  if ( v19 == *v9 )
    goto LABEL_3;
  v15 = 1;
  v16 = 0;
  while ( v10 != -8 )
  {
    if ( !v16 && v10 == -16 )
      v16 = v9;
    v8 = (v5 - 1) & (v15 + v8);
    v9 = (__int64 *)(v7 + ((unsigned __int64)v8 << 6));
    v10 = *v9;
    if ( v19 == *v9 )
      goto LABEL_3;
    ++v15;
  }
  v17 = *(_DWORD *)(a1 + 1944);
  if ( v16 )
    v9 = v16;
  ++*(_QWORD *)(a1 + 1928);
  v18 = v17 + 1;
  if ( 4 * (v17 + 1) >= 3 * v5 )
    goto LABEL_17;
  if ( v5 - *(_DWORD *)(a1 + 1948) - v18 <= v5 >> 3 )
  {
    sub_19EE3D0(v3, v5);
    goto LABEL_18;
  }
LABEL_13:
  *(_DWORD *)(a1 + 1944) = v18;
  if ( *v9 != -8 )
    --*(_DWORD *)(a1 + 1948);
  *v9 = v6;
  v9[1] = 0;
  v9[2] = (__int64)(v9 + 6);
  v9[3] = (__int64)(v9 + 6);
  v9[4] = 2;
  *((_DWORD *)v9 + 10) = 0;
LABEL_3:
  v11 = sub_1412190((__int64)(v9 + 1), a3);
  v12 = v9[3];
  if ( v12 == v9[2] )
    v13 = *((unsigned int *)v9 + 9);
  else
    v13 = *((unsigned int *)v9 + 8);
  v20[0] = v11;
  v20[1] = v12 + 8 * v13;
  return sub_19E4730((__int64)v20);
}
