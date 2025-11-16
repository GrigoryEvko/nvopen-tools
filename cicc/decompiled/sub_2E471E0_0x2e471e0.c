// Function: sub_2E471E0
// Address: 0x2e471e0
//
__int64 __fastcall sub_2E471E0(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, __int64 a5, char a6)
{
  __int64 v6; // r10
  __int64 v9; // rsi
  __int64 v10; // rdx
  int v11; // edi
  unsigned int v13; // ecx
  int *v14; // rax
  int v15; // r12d
  __int64 v16; // r12
  unsigned int v17; // r13d
  __int64 v18; // r8
  __int64 v19; // rax
  __int64 v20; // rdx
  int v21; // ecx
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rax
  __int16 *v26; // rax
  __int16 *v27; // rax
  int v28; // r14d
  int v29; // eax
  int v30; // r13d
  __int64 v31; // [rsp+8h] [rbp-98h]
  int v32; // [rsp+1Ch] [rbp-84h] BYREF
  __int64 v33[4]; // [rsp+20h] [rbp-80h] BYREF
  int v34; // [rsp+40h] [rbp-60h] BYREF
  __int16 *v35; // [rsp+48h] [rbp-58h]
  __int16 v36; // [rsp+50h] [rbp-50h]
  int v37; // [rsp+58h] [rbp-48h]
  __int64 v38; // [rsp+60h] [rbp-40h]
  __int16 v39; // [rsp+68h] [rbp-38h]

  v6 = 24LL * a3;
  v9 = *(_QWORD *)(a1 + 8);
  v10 = *(unsigned int *)(a1 + 24);
  v11 = *(_DWORD *)(*(_QWORD *)(a4 + 8) + v6 + 16) & 0xFFF;
  if ( !(_DWORD)v10 )
    return 0;
  v13 = (v10 - 1) & (37 * v11);
  v14 = (int *)(v9 + ((unsigned __int64)v13 << 7));
  v15 = *v14;
  if ( v11 != *v14 )
  {
    v29 = 1;
    while ( v15 != -1 )
    {
      v30 = v29 + 1;
      v13 = (v10 - 1) & (v29 + v13);
      v14 = (int *)(v9 + ((unsigned __int64)v13 << 7));
      v15 = *v14;
      if ( v11 == *v14 )
        goto LABEL_3;
      v29 = v30;
    }
    return 0;
  }
LABEL_3:
  if ( v14 == (int *)(v9 + (v10 << 7)) )
    return 0;
  v31 = v6;
  if ( !*((_BYTE *)v14 + 120) )
    return 0;
  v16 = *((_QWORD *)v14 + 1);
  sub_2E44C10((__int64)v33, v16, a5, a6);
  v17 = *(_DWORD *)(v33[0] + 8);
  if ( a3 != v17 )
  {
    v23 = *(_QWORD *)(a4 + 8);
    v32 = *(_DWORD *)(v33[0] + 8);
    v37 = 0;
    v24 = *(unsigned int *)(v23 + v31 + 8);
    v25 = *(_QWORD *)(a4 + 56);
    v38 = 0;
    v26 = (__int16 *)(v25 + 2 * v24);
    LODWORD(v24) = *v26;
    v27 = v26 + 1;
    v28 = v24 + a3;
    if ( !(_WORD)v24 )
      v27 = 0;
    v34 = v28;
    v36 = v28;
    v35 = v27;
    v39 = 0;
    if ( !sub_2E46590(&v34, &v32) )
      return 0;
  }
  v18 = v16;
  if ( a2 != v16 )
  {
    while ( 1 )
    {
      v19 = *(_QWORD *)(v18 + 32);
      v20 = v19 + 40LL * (*(_DWORD *)(v18 + 40) & 0xFFFFFF);
      if ( v19 != v20 )
        break;
LABEL_14:
      v18 = *(_QWORD *)(v18 + 8);
      if ( a2 == v18 )
        return v16;
    }
    while ( 1 )
    {
      if ( *(_BYTE *)v19 == 12 )
      {
        v21 = *(_DWORD *)(*(_QWORD *)(v19 + 24) + 4LL * (v17 >> 5));
        if ( !_bittest(&v21, v17 & 0x1F) )
          return 0;
      }
      v19 += 40;
      if ( v20 == v19 )
        goto LABEL_14;
    }
  }
  return v16;
}
