// Function: sub_33CD850
// Address: 0x33cd850
//
__int64 __fastcall sub_33CD850(__int64 a1, __int64 a2, unsigned __int64 a3, char a4)
{
  __int64 *v6; // rdi
  __int64 v7; // r14
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rsi
  unsigned int v13; // r12d
  __int64 (*v14)(); // rax
  unsigned int v15; // r15d
  __int64 v17; // rsi
  __int64 v18; // rdi
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r9
  __int64 v22; // rax
  unsigned int v23; // eax
  __int64 v24; // rdx
  __int64 v25; // [rsp+10h] [rbp-60h] BYREF
  unsigned __int64 v26; // [rsp+18h] [rbp-58h]
  unsigned __int16 v27; // [rsp+2Ah] [rbp-46h] BYREF
  unsigned int v28; // [rsp+2Ch] [rbp-44h] BYREF
  __int64 v29[8]; // [rsp+30h] [rbp-40h] BYREF

  v6 = *(__int64 **)(a1 + 40);
  v25 = a2;
  v26 = a3;
  v7 = sub_2E79000(v6);
  v12 = sub_3007410((__int64)&v25, *(__int64 **)(a1 + 64), v8, v9, v10, v11);
  if ( a4 )
    v13 = sub_AE5020(v7, v12);
  else
    v13 = sub_AE5260(v7, v12);
  if ( (_WORD)v25 )
  {
    if ( *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL * (unsigned __int16)v25 + 112) || (unsigned __int16)(v25 - 17) > 0xD3u )
      return v13;
  }
  else if ( !sub_30070B0((__int64)&v25) )
  {
    return v13;
  }
  v14 = *(__int64 (**)())(**(_QWORD **)(*(_QWORD *)(a1 + 40) + 16LL) + 136LL);
  if ( v14 == sub_2DD19D0 )
    BUG();
  v15 = *(unsigned __int8 *)(v14() + 12);
  if ( (unsigned __int8)v15 < (unsigned __int8)v13 )
  {
    LOWORD(v29[0]) = 0;
    v17 = *(_QWORD *)(a1 + 64);
    v18 = *(_QWORD *)(a1 + 16);
    v27 = 0;
    v29[1] = 0;
    sub_2FE8D10(v18, v17, (unsigned int)v25, v26, v29, &v28, &v27);
    v22 = sub_3007410((__int64)v29, *(__int64 **)(a1 + 64), v19, v20, (__int64)v29, v21);
    if ( a4 )
      v23 = sub_AE5020(v7, v22);
    else
      v23 = sub_AE5260(v7, v22);
    v24 = *(_QWORD *)(a1 + 40);
    if ( (unsigned __int8)v23 >= (unsigned __int8)v13 )
    {
      if ( !*(_BYTE *)(*(_QWORD *)(v24 + 48) + 1LL) )
        return v15;
    }
    else
    {
      v13 = v23;
      if ( !*(_BYTE *)(*(_QWORD *)(v24 + 48) + 1LL) && (unsigned __int8)v15 <= (unsigned __int8)v23 )
        return v15;
    }
  }
  return v13;
}
