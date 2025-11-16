// Function: sub_32CB9C0
// Address: 0x32cb9c0
//
__int64 __fastcall sub_32CB9C0(__int64 a1, _QWORD *a2)
{
  unsigned __int16 *v2; // rax
  __int64 v3; // rdx
  __int64 v4; // r14
  __int64 v5; // rax
  int v6; // r15d
  __int64 result; // rax
  _DWORD *v8; // rax
  _QWORD *v10; // r9
  __int64 v11; // rdi
  int v12; // ecx
  __int64 v13; // rax
  int v14; // esi
  __int64 v15; // r10
  bool (__fastcall *v16)(__int64, __int64, unsigned __int16); // rax
  __int64 v17; // rax
  __int64 v18; // r12
  __int64 v19; // r13
  _QWORD *v20; // r9
  __int64 v21; // rsi
  __int128 v22; // rax
  int v23; // r9d
  __int64 v24; // r9
  __int64 v25; // r12
  __int64 v26; // rdx
  __int64 v27; // r13
  __int64 v28; // r8
  __int64 v29; // r8
  __int64 v30; // rax
  __int64 v31; // rax
  __int128 v32; // [rsp-20h] [rbp-90h]
  __int128 v33; // [rsp-10h] [rbp-80h]
  __int128 v34; // [rsp+0h] [rbp-70h]
  _QWORD *v35; // [rsp+18h] [rbp-58h]
  __int64 v36; // [rsp+18h] [rbp-58h]
  _QWORD *v37; // [rsp+18h] [rbp-58h]
  __int64 v38; // [rsp+28h] [rbp-48h] BYREF
  __int64 v39; // [rsp+30h] [rbp-40h] BYREF
  int v40; // [rsp+38h] [rbp-38h]

  v2 = (unsigned __int16 *)a2[6];
  v3 = *v2;
  v4 = *((_QWORD *)v2 + 1);
  v5 = a2[7];
  v6 = (unsigned __int16)v3;
  if ( !v5 )
    return 0;
  if ( *(_QWORD *)(v5 + 32) )
    return 0;
  v8 = (_DWORD *)a2[5];
  v10 = a2;
  v11 = *(_QWORD *)v8;
  v12 = v8[2];
  v13 = *(_QWORD *)(*(_QWORD *)v8 + 56LL);
  if ( !v13 )
    return 0;
  v14 = 1;
  do
  {
    if ( v12 == *(_DWORD *)(v13 + 8) )
    {
      if ( !v14 )
        return 0;
      v13 = *(_QWORD *)(v13 + 32);
      if ( !v13 )
        goto LABEL_14;
      if ( v12 == *(_DWORD *)(v13 + 8) )
        return 0;
      v14 = 0;
    }
    v13 = *(_QWORD *)(v13 + 32);
  }
  while ( v13 );
  if ( v14 == 1 )
    return 0;
LABEL_14:
  v15 = *(_QWORD *)(a1 + 8);
  v16 = *(bool (__fastcall **)(__int64, __int64, unsigned __int16))(*(_QWORD *)v15 + 2192LL);
  if ( v16 == sub_302E170 )
  {
    if ( !(_WORD)v3 || !*(_QWORD *)(v15 + 8 * v3 + 112) )
      return 0;
  }
  else
  {
    v37 = v10;
    if ( !((unsigned __int8 (__fastcall *)(__int64, __int64, _QWORD, __int64))v16)(v15, 186, (unsigned __int16)v3, v4) )
      return 0;
    v10 = v37;
    v11 = *(_QWORD *)v37[5];
  }
  v17 = *(_QWORD *)(v11 + 40);
  v35 = v10;
  v18 = *(_QWORD *)(v17 + 40);
  v19 = *(_QWORD *)(v17 + 48);
  if ( !(unsigned __int8)sub_326A930(v18, v19, 1u) )
    return 0;
  v20 = v35;
  v21 = v35[10];
  v39 = v21;
  if ( v21 )
  {
    sub_B96E90((__int64)&v39, v21, 1);
    v20 = v35;
  }
  v40 = *((_DWORD *)v20 + 18);
  *(_QWORD *)&v22 = sub_33FAF80(
                      *(_QWORD *)a1,
                      216,
                      (unsigned int)&v39,
                      v6,
                      v4,
                      (_DWORD)v20,
                      *(_OWORD *)*(_QWORD *)(*(_QWORD *)v20[5] + 40LL));
  *((_QWORD *)&v32 + 1) = v19;
  *(_QWORD *)&v32 = v18;
  v34 = v22;
  v25 = sub_33FAF80(*(_QWORD *)a1, 216, (unsigned int)&v39, v6, v4, v23, v32);
  v27 = v26;
  if ( *(_DWORD *)(v34 + 24) != 328 )
  {
    v38 = v34;
    sub_32B3B20(a1 + 568, &v38);
    v24 = v34;
    if ( *(int *)(v34 + 88) < 0 )
    {
      *(_DWORD *)(v34 + 88) = *(_DWORD *)(a1 + 48);
      v30 = *(unsigned int *)(a1 + 48);
      if ( v30 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 52) )
      {
        sub_C8D5F0(a1 + 40, (const void *)(a1 + 56), v30 + 1, 8u, v28, v34);
        v30 = *(unsigned int *)(a1 + 48);
        v24 = v34;
      }
      *(_QWORD *)(*(_QWORD *)(a1 + 40) + 8 * v30) = v24;
      ++*(_DWORD *)(a1 + 48);
    }
  }
  if ( *(_DWORD *)(v25 + 24) != 328 )
  {
    v38 = v25;
    sub_32B3B20(a1 + 568, &v38);
    v24 = v25;
    if ( *(int *)(v25 + 88) < 0 )
    {
      *(_DWORD *)(v25 + 88) = *(_DWORD *)(a1 + 48);
      v31 = *(unsigned int *)(a1 + 48);
      if ( v31 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 52) )
      {
        sub_C8D5F0(a1 + 40, (const void *)(a1 + 56), v31 + 1, 8u, v29, v25);
        v31 = *(unsigned int *)(a1 + 48);
        v24 = v25;
      }
      *(_QWORD *)(*(_QWORD *)(a1 + 40) + 8 * v31) = v24;
      ++*(_DWORD *)(a1 + 48);
    }
  }
  *((_QWORD *)&v33 + 1) = v27;
  *(_QWORD *)&v33 = v25;
  result = sub_3406EB0(*(_QWORD *)a1, 186, (unsigned int)&v39, v6, v4, v24, v34, v33);
  if ( v39 )
  {
    v36 = result;
    sub_B91220((__int64)&v39, v39);
    return v36;
  }
  return result;
}
