// Function: sub_181D080
// Address: 0x181d080
//
__int64 *__fastcall sub_181D080(__int64 *a1, __int64 a2)
{
  __int64 v3; // r12
  __int64 v5; // r13
  _QWORD *v6; // rax
  unsigned __int8 v7; // dl
  __int64 v8; // rbx
  __int64 v9; // r13
  unsigned int v10; // esi
  __int64 v11; // rcx
  unsigned int v12; // edx
  __int64 *result; // rax
  __int64 v14; // r8
  __int64 v15; // rax
  __int64 *v16; // r14
  __int64 v17; // rax
  char v18; // al
  unsigned int v19; // esi
  int v20; // eax
  __int64 v21; // rdi
  int v22; // eax
  __int64 v23; // rax
  unsigned int v24; // edx
  _QWORD *v25; // rax
  __int64 v26; // r14
  __int64 v27; // rsi
  __int64 v28; // rdx
  int v29; // r11d
  __int64 *v30; // r9
  int v31; // ecx
  int v32; // ecx
  unsigned int v33; // [rsp+18h] [rbp-E8h]
  _QWORD *v34; // [rsp+20h] [rbp-E0h]
  __int64 *v35; // [rsp+20h] [rbp-E0h]
  __int64 *v36; // [rsp+28h] [rbp-D8h]
  __int64 v37; // [rsp+38h] [rbp-C8h] BYREF
  __int64 v38; // [rsp+40h] [rbp-C0h] BYREF
  __int16 v39; // [rsp+50h] [rbp-B0h]
  __int64 v40[2]; // [rsp+60h] [rbp-A0h] BYREF
  __int16 v41; // [rsp+70h] [rbp-90h]
  __int64 *v42; // [rsp+80h] [rbp-80h] BYREF
  __int64 v43; // [rsp+88h] [rbp-78h]
  __int64 *v44; // [rsp+90h] [rbp-70h]

  v3 = a2;
  v5 = *(_QWORD *)(a2 + 8);
  if ( !v5 )
  {
LABEL_11:
    sub_17CE510((__int64)&v42, a2, 0, 0, 0);
    v16 = (__int64 *)*a1;
    v39 = 257;
    v17 = *v16;
    v37 = a2;
    v34 = *(_QWORD **)(v17 + 176);
    v18 = sub_1819C90((__int64)(v16 + 20), &v37, v40);
    v36 = (__int64 *)v40[0];
    if ( v18 )
    {
LABEL_17:
      v24 = *(_DWORD *)(sub_1632FA0(*(_QWORD *)(*(_QWORD *)(v43 + 56) + 40LL)) + 4);
      v41 = 257;
      v33 = v24;
      v25 = sub_1648A60(64, 1u);
      v26 = (__int64)v25;
      if ( v25 )
        sub_15F8BC0((__int64)v25, v34, v33, 0, (__int64)v40, 0);
      if ( v43 )
      {
        v35 = v44;
        sub_157E9D0(v43 + 40, v26);
        v27 = *v35;
        v28 = *(_QWORD *)(v26 + 24) & 7LL;
        *(_QWORD *)(v26 + 32) = v35;
        v27 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v26 + 24) = v27 | v28;
        *(_QWORD *)(v27 + 8) = v26 + 24;
        *v35 = *v35 & 7 | (v26 + 24);
      }
      sub_164B780(v26, &v38);
      sub_12A86E0((__int64 *)&v42, v26);
      v36[1] = v26;
      sub_17CD270((__int64 *)&v42);
      goto LABEL_5;
    }
    v19 = *((_DWORD *)v16 + 46);
    v20 = *((_DWORD *)v16 + 44);
    ++v16[20];
    v21 = (__int64)(v16 + 20);
    v22 = v20 + 1;
    if ( 4 * v22 >= 3 * v19 )
    {
      v19 *= 2;
    }
    else if ( v19 - *((_DWORD *)v16 + 45) - v22 > v19 >> 3 )
    {
LABEL_14:
      *((_DWORD *)v16 + 44) = v22;
      if ( *v36 != -8 )
        --*((_DWORD *)v16 + 45);
      v23 = v37;
      v36[1] = 0;
      *v36 = v23;
      goto LABEL_17;
    }
    sub_181CEC0(v21, v19);
    sub_1819C90(v21, &v37, v40);
    v36 = (__int64 *)v40[0];
    v22 = *((_DWORD *)v16 + 44) + 1;
    goto LABEL_14;
  }
  while ( 1 )
  {
    v6 = sub_1648700(v5);
    v7 = *((_BYTE *)v6 + 16);
    if ( v7 <= 0x17u )
      break;
    if ( v7 != 54 )
    {
      if ( v7 != 55 )
        break;
      v15 = *(v6 - 3);
      if ( !v15 || v15 != a2 )
        break;
    }
    v5 = *(_QWORD *)(v5 + 8);
    if ( !v5 )
      goto LABEL_11;
  }
LABEL_5:
  v8 = *a1;
  v9 = *(_QWORD *)(*(_QWORD *)v8 + 200LL);
  v40[0] = a2;
  v10 = *(_DWORD *)(v8 + 152);
  if ( !v10 )
  {
    ++*(_QWORD *)(v8 + 128);
LABEL_32:
    v10 *= 2;
    goto LABEL_33;
  }
  v11 = *(_QWORD *)(v8 + 136);
  v12 = (v10 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  result = (__int64 *)(v11 + 16LL * v12);
  v14 = *result;
  if ( *result == a2 )
    goto LABEL_7;
  v29 = 1;
  v30 = 0;
  while ( v14 != -8 )
  {
    if ( !v30 && v14 == -16 )
      v30 = result;
    v12 = (v10 - 1) & (v29 + v12);
    result = (__int64 *)(v11 + 16LL * v12);
    v14 = *result;
    if ( *result == a2 )
      goto LABEL_7;
    ++v29;
  }
  v31 = *(_DWORD *)(v8 + 144);
  if ( v30 )
    result = v30;
  ++*(_QWORD *)(v8 + 128);
  v32 = v31 + 1;
  if ( 4 * v32 >= 3 * v10 )
    goto LABEL_32;
  if ( v10 - *(_DWORD *)(v8 + 148) - v32 <= v10 >> 3 )
  {
LABEL_33:
    sub_176F940(v8 + 128, v10);
    sub_176A9A0(v8 + 128, v40, &v42);
    result = v42;
    v3 = v40[0];
    v32 = *(_DWORD *)(v8 + 144) + 1;
  }
  *(_DWORD *)(v8 + 144) = v32;
  if ( *result != -8 )
    --*(_DWORD *)(v8 + 148);
  *result = v3;
  result[1] = 0;
LABEL_7:
  result[1] = v9;
  return result;
}
