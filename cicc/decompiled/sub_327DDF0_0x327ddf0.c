// Function: sub_327DDF0
// Address: 0x327ddf0
//
__int64 __fastcall sub_327DDF0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8)
{
  __int64 v8; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // r14
  __int64 *v16; // rax
  __int64 v17; // rbx
  __int64 *v18; // rax
  __int64 v19; // r14
  __int64 v20; // rax
  __int64 v21; // r8
  __int64 v22; // rdx
  __int64 v23; // rax
  __int64 v24; // r9
  __int128 v25; // rax
  __int128 v26; // rax
  __int64 v27; // [rsp+0h] [rbp-60h]
  __int128 v28; // [rsp+0h] [rbp-60h]
  __int64 v30; // [rsp+18h] [rbp-48h]
  __int64 v31; // [rsp+18h] [rbp-48h]
  __int64 v32; // [rsp+18h] [rbp-48h]
  __int64 v33; // [rsp+20h] [rbp-40h] BYREF
  int v34; // [rsp+28h] [rbp-38h]

  v8 = 1;
  if ( (_WORD)a7 != 1 )
  {
    if ( !(_WORD)a7 )
      return 0;
    v8 = (unsigned __int16)a7;
    if ( !*(_QWORD *)(a1 + 8LL * (unsigned __int16)a7 + 112) )
      return 0;
  }
  if ( (*(_BYTE *)(a1 + 500 * v8 + 6608) & 0xFB) != 0 )
    return 0;
  if ( *(_DWORD *)(a4 + 24) != 186 )
    return 0;
  if ( *(_DWORD *)(a5 + 24) != 186 )
    return 0;
  v12 = *(_QWORD *)(a4 + 56);
  if ( !v12 )
    return 0;
  if ( *(_QWORD *)(v12 + 32) )
    return 0;
  v13 = *(_QWORD *)(a5 + 56);
  if ( !v13 )
    return 0;
  if ( *(_QWORD *)(v13 + 32) )
    return 0;
  v27 = sub_33DFBC0(*(_QWORD *)(*(_QWORD *)(a4 + 40) + 40LL), *(_QWORD *)(*(_QWORD *)(a4 + 40) + 48LL), 0, 0);
  v14 = sub_33DFBC0(*(_QWORD *)(*(_QWORD *)(a5 + 40) + 40LL), *(_QWORD *)(*(_QWORD *)(a5 + 40) + 48LL), 0, 0);
  v15 = v14;
  if ( !v27 )
    return 0;
  if ( !v14 )
    return 0;
  if ( !sub_D94970(*(_QWORD *)(v27 + 96) + 24LL, (_QWORD *)0xFF00FF00LL) )
    return 0;
  if ( !sub_D94970(*(_QWORD *)(v15 + 96) + 24LL, &byte_FF00FF) )
    return 0;
  v16 = *(__int64 **)(a4 + 40);
  v17 = *v16;
  if ( *(_DWORD *)(*v16 + 24) != 190 )
    return 0;
  v18 = *(__int64 **)(a5 + 40);
  v19 = *v18;
  if ( *(_DWORD *)(*v18 + 24) != 192 )
    return 0;
  v30 = sub_33DFBC0(*(_QWORD *)(*(_QWORD *)(v17 + 40) + 40LL), *(_QWORD *)(*(_QWORD *)(v17 + 40) + 48LL), 0, 0);
  v20 = sub_33DFBC0(*(_QWORD *)(*(_QWORD *)(v19 + 40) + 40LL), *(_QWORD *)(*(_QWORD *)(v19 + 40) + 48LL), 0, 0);
  v21 = v30;
  if ( !v30 )
    return 0;
  v31 = v20;
  if ( !v20 )
    return 0;
  if ( !sub_D94970(*(_QWORD *)(v21 + 96) + 24LL, (_QWORD *)8) )
    return 0;
  if ( !sub_D94970(*(_QWORD *)(v31 + 96) + 24LL, (_QWORD *)8) )
    return 0;
  v22 = *(_QWORD *)(v19 + 40);
  v23 = *(_QWORD *)(v17 + 40);
  v24 = a8;
  if ( *(_QWORD *)v23 != *(_QWORD *)v22 || *(_DWORD *)(v23 + 8) != *(_DWORD *)(v22 + 8) )
    return 0;
  v33 = *(_QWORD *)(a3 + 80);
  if ( v33 )
  {
    sub_325F5D0(&v33);
    v23 = *(_QWORD *)(v17 + 40);
    v24 = a8;
  }
  v32 = v24;
  v34 = *(_DWORD *)(a3 + 72);
  *(_QWORD *)&v25 = sub_33FAF80(a2, 197, (unsigned int)&v33, a7, v24, v24, *(_OWORD *)v23);
  v28 = v25;
  *(_QWORD *)&v26 = sub_3400E40(a2, 16, (unsigned int)a7, v32, &v33);
  *(_QWORD *)&v28 = sub_3406EB0(a2, 194, (unsigned int)&v33, a7, v32, v32, v28, v26);
  sub_9C6650(&v33);
  return v28;
}
