// Function: sub_3034FC0
// Address: 0x3034fc0
//
__int64 __fastcall sub_3034FC0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  unsigned int *v5; // rdx
  __int64 v6; // r12
  int v7; // eax
  unsigned __int16 *v8; // rax
  __int64 v9; // r10
  int v10; // r13d
  __int64 i; // rax
  __int64 v12; // rsi
  __int64 v13; // rdi
  __int128 v14; // rax
  int v15; // r9d
  __int64 v16; // r14
  int v17; // r9d
  int v18; // r9d
  int v19; // edx
  __int128 v20; // [rsp-78h] [rbp-78h]
  __int64 v21; // [rsp-78h] [rbp-78h]
  int v22; // [rsp-60h] [rbp-60h]
  int v23; // [rsp-58h] [rbp-58h]
  __int64 v24; // [rsp-58h] [rbp-58h]
  __int64 v25; // [rsp-58h] [rbp-58h]
  __int64 v26; // [rsp-48h] [rbp-48h] BYREF
  int v27; // [rsp-40h] [rbp-40h]

  if ( *(int *)(a2 + 8) > 1 )
    return 0;
  v5 = *(unsigned int **)(a1 + 40);
  v6 = *(_QWORD *)v5;
  v7 = *(_DWORD *)(*(_QWORD *)v5 + 24LL);
  if ( v7 > 58 )
  {
    if ( (unsigned int)(v7 - 186) > 2 )
      return 0;
  }
  else if ( v7 <= 55 )
  {
    return 0;
  }
  v8 = *(unsigned __int16 **)(a1 + 48);
  v9 = *((_QWORD *)v8 + 1);
  v10 = *v8;
  if ( *(_WORD *)(*(_QWORD *)(v6 + 48) + 16LL * v5[2]) != 8 || *v8 != 7 )
    return 0;
  for ( i = *(_QWORD *)(v6 + 56); i; i = *(_QWORD *)(i + 32) )
  {
    v19 = *(_DWORD *)(*(_QWORD *)(i + 16) + 24LL);
    if ( v19 >= 0 )
    {
      if ( v19 != 216 )
        return 0;
    }
    else if ( v19 != -1206 )
    {
      return 0;
    }
  }
  v12 = *(_QWORD *)(a1 + 80);
  v26 = v12;
  if ( v12 )
  {
    v23 = v9;
    sub_B96E90((__int64)&v26, v12, 1);
    LODWORD(v9) = v23;
  }
  v13 = *(_QWORD *)(a2 + 16);
  v22 = v9;
  v27 = *(_DWORD *)(a1 + 72);
  *(_QWORD *)&v14 = sub_3400BD0(v13, 0, (unsigned int)&v26, 7, 0, 1, 0);
  *((_QWORD *)&v20 + 1) = *((_QWORD *)&v14 + 1);
  v24 = v14;
  v16 = sub_33F77A0(*(_QWORD *)(a2 + 16), 1205, (unsigned int)&v26, v10, v22, v15, *(_OWORD *)*(_QWORD *)(v6 + 40), v14);
  *(_QWORD *)&v20 = v24;
  v21 = sub_33F77A0(
          *(_QWORD *)(a2 + 16),
          1205,
          (unsigned int)&v26,
          v10,
          v22,
          v17,
          *(_OWORD *)(*(_QWORD *)(v6 + 40) + 40LL),
          v20);
  result = sub_3406EB0(
             *(_QWORD *)(a2 + 16),
             *(_DWORD *)(v6 + 24),
             (unsigned int)&v26,
             v10,
             v22,
             v18,
             (unsigned __int64)v16,
             (unsigned __int64)v21);
  if ( v26 )
  {
    v25 = result;
    sub_B91220((__int64)&v26, v26);
    return v25;
  }
  return result;
}
