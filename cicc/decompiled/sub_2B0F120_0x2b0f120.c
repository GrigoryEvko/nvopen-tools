// Function: sub_2B0F120
// Address: 0x2b0f120
//
__int64 __fastcall sub_2B0F120(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5)
{
  __int64 v5; // r11
  __int64 v6; // r9
  unsigned int v8; // edx
  int v9; // r15d
  unsigned int v10; // r13d
  __int64 v11; // rbx
  __int64 v12; // r14
  unsigned int v13; // r8d
  __int64 v14; // rcx
  __int64 v15; // rax
  int v16; // edi
  int v17; // edx
  unsigned __int64 v19; // rcx
  __int64 *v20; // [rsp+0h] [rbp-60h]
  __int64 v21; // [rsp+10h] [rbp-50h]
  unsigned int v22; // [rsp+20h] [rbp-40h]
  int v23; // [rsp+24h] [rbp-3Ch]
  unsigned __int64 v24; // [rsp+28h] [rbp-38h]

  v5 = a3;
  v6 = a5;
  if ( *(_BYTE *)(a2 + 8) != 17 )
    return sub_DFAAD0(a1, a3, a4, 1u, 0);
  v8 = *(_DWORD *)(a4 + 8);
  v9 = *(_DWORD *)(a2 + 32);
  if ( !v8 )
    return 0;
  v23 = 0;
  v10 = 0;
  v11 = 0;
  v24 = 0;
  v12 = v8 - 1LL;
  v13 = *(_DWORD *)(a4 + 8);
  while ( 1 )
  {
    v14 = *(_QWORD *)a4;
    if ( v13 > 0x40 )
      v14 = *(_QWORD *)(v14 + 8LL * ((unsigned int)v11 >> 6));
    if ( (v14 & (1LL << v11)) != 0 )
    {
      v22 = v6;
      v21 = v5;
      v20 = a1;
      v15 = sub_DFBC30(a1, 4, v5, 0, 0, v6, v10, a2, 0, 0, 0);
      v16 = 1;
      if ( v17 != 1 )
        v16 = v23;
      v5 = v21;
      v6 = v22;
      v23 = v16;
      a1 = v20;
      if ( __OFADD__(v15, v24) )
      {
        v19 = 0x8000000000000000LL;
        if ( v15 > 0 )
          v19 = 0x7FFFFFFFFFFFFFFFLL;
        v24 = v19;
      }
      else
      {
        v24 += v15;
      }
    }
    v10 += v9;
    if ( v12 == v11 )
      break;
    v13 = *(_DWORD *)(a4 + 8);
    ++v11;
  }
  return v24;
}
