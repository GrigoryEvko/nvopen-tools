// Function: sub_328C120
// Address: 0x328c120
//
__int64 __fastcall sub_328C120(
        __int64 *a1,
        unsigned int a2,
        unsigned int a3,
        int a4,
        int a5,
        int a6,
        __int64 a7,
        __int64 a8,
        int a9)
{
  __int64 result; // rax
  __int64 v12; // rax
  __int64 v13; // rdx
  unsigned __int16 v14; // r11
  __int64 v15; // rcx
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rdx
  unsigned int *v21; // rdx
  __int64 v22; // rbx
  __int128 v23; // rax
  unsigned __int16 v25; // [rsp+Ch] [rbp-64h]
  __int64 v26; // [rsp+10h] [rbp-60h]
  __int64 v27; // [rsp+18h] [rbp-58h]
  __int64 v28; // [rsp+20h] [rbp-50h] BYREF
  int v29; // [rsp+28h] [rbp-48h]
  __int64 v30; // [rsp+30h] [rbp-40h]

  if ( *(_DWORD *)(a7 + 24) != a2 )
    return 0;
  if ( a2 != *(_DWORD *)(a8 + 24) )
    return 0;
  v12 = *(_QWORD *)(**(_QWORD **)(a8 + 40) + 48LL) + 16LL * *(unsigned int *)(*(_QWORD *)(a8 + 40) + 8LL);
  v13 = *(_QWORD *)(**(_QWORD **)(a7 + 40) + 48LL) + 16LL * *(unsigned int *)(*(_QWORD *)(a7 + 40) + 8LL);
  v14 = *(_WORD *)v13;
  if ( *(_WORD *)v12 != *(_WORD *)v13 )
    return 0;
  v15 = *(_QWORD *)(v13 + 8);
  if ( *(_QWORD *)(v12 + 8) != v15 && !v14 )
    return 0;
  v17 = *(_QWORD *)(a7 + 56);
  if ( !v17 )
    return 0;
  if ( *(_QWORD *)(v17 + 32) )
    return 0;
  v18 = *(_QWORD *)(a8 + 56);
  if ( !v18 )
    return 0;
  if ( *(_QWORD *)(v18 + 32) )
    return 0;
  v25 = *(_WORD *)v13;
  v26 = *(_QWORD *)(v13 + 8);
  v27 = a1[1];
  if ( !(unsigned __int8)sub_328A020(v27, a3, v14, v15, 0)
    || !(*(unsigned __int8 (__fastcall **)(__int64, _QWORD, _QWORD, __int64))(*(_QWORD *)v27 + 152LL))(
          v27,
          a2,
          v25,
          v26) )
  {
    return 0;
  }
  v19 = *a1;
  v29 = a9;
  v20 = *(_QWORD *)(v19 + 1024);
  v28 = v19;
  v30 = v20;
  *(_QWORD *)(v19 + 1024) = &v28;
  v21 = *(unsigned int **)(a7 + 40);
  v22 = *a1;
  *(_QWORD *)&v23 = sub_3406EB0(
                      *a1,
                      a3,
                      a4,
                      *(unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)v21 + 48LL) + 16LL * v21[2]),
                      *(_QWORD *)(*(_QWORD *)(*(_QWORD *)v21 + 48LL) + 16LL * v21[2] + 8),
                      a6,
                      *(_OWORD *)v21,
                      *(_OWORD *)*(_QWORD *)(a8 + 40));
  result = sub_33FAF80(v22, a2, a4, a5, a6, a6, v23);
  *(_QWORD *)(v28 + 1024) = v30;
  return result;
}
