// Function: sub_2E465E0
// Address: 0x2e465e0
//
__int64 __fastcall sub_2E465E0(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, __int64 a5, char a6)
{
  __int64 v6; // r15
  __int64 v9; // rsi
  __int64 v10; // rdx
  int v11; // edi
  unsigned int v13; // ecx
  int *v14; // rax
  int v15; // r11d
  __int64 v16; // r12
  unsigned int v17; // r10d
  unsigned int v18; // edx
  __int64 v19; // r11
  unsigned int v20; // r8d
  unsigned int v21; // r10d
  unsigned int v22; // edi
  __int64 v23; // r8
  __int64 v24; // r9
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rcx
  int v28; // esi
  int v29; // ecx
  __int64 v31; // rax
  __int64 v32; // rcx
  __int64 v33; // rax
  __int16 *v34; // rax
  __int16 *v35; // rax
  int v36; // r13d
  bool v37; // al
  int v38; // eax
  int v39; // r12d
  unsigned int v40; // [rsp+8h] [rbp-98h]
  unsigned int v41; // [rsp+Ch] [rbp-94h]
  int v42; // [rsp+1Ch] [rbp-84h] BYREF
  __int64 v43; // [rsp+20h] [rbp-80h] BYREF
  __int64 v44; // [rsp+28h] [rbp-78h]
  int v45; // [rsp+40h] [rbp-60h] BYREF
  __int16 *v46; // [rsp+48h] [rbp-58h]
  __int16 v47; // [rsp+50h] [rbp-50h]
  int v48; // [rsp+58h] [rbp-48h]
  __int64 v49; // [rsp+60h] [rbp-40h]
  __int16 v50; // [rsp+68h] [rbp-38h]

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
    v38 = 1;
    while ( v15 != -1 )
    {
      v39 = v38 + 1;
      v13 = (v10 - 1) & (v38 + v13);
      v14 = (int *)(v9 + ((unsigned __int64)v13 << 7));
      v15 = *v14;
      if ( v11 == *v14 )
        goto LABEL_3;
      v38 = v39;
    }
    return 0;
  }
LABEL_3:
  if ( v14 == (int *)(v9 + (v10 << 7)) )
    return 0;
  if ( !*((_BYTE *)v14 + 120) )
    return 0;
  v16 = *((_QWORD *)v14 + 1);
  if ( !v16 )
    return 0;
  sub_2E44C10((__int64)&v43, *((_QWORD *)v14 + 1), a5, a6);
  v17 = *(_DWORD *)(v44 + 8);
  v18 = *(_DWORD *)(v43 + 8);
  if ( a3 != v18 )
  {
    v31 = *(_QWORD *)(a4 + 8);
    v40 = *(_DWORD *)(v44 + 8);
    v42 = *(_DWORD *)(v43 + 8);
    v32 = *(unsigned int *)(v31 + v6 + 8);
    v33 = *(_QWORD *)(a4 + 56);
    v41 = v18;
    v48 = 0;
    v34 = (__int16 *)(v33 + 2 * v32);
    v49 = 0;
    LODWORD(v32) = *v34;
    v35 = v34 + 1;
    v36 = v32 + a3;
    if ( !(_WORD)v32 )
      v35 = 0;
    v45 = v36;
    v47 = v36;
    v46 = v35;
    v50 = 0;
    v37 = sub_2E46590(&v45, &v42);
    v18 = v41;
    v17 = v40;
    if ( !v37 )
      return 0;
  }
  v19 = v16;
  if ( a2 != v16 )
  {
    v20 = v17;
    v21 = v17 & 0x1F;
    v22 = v18 & 0x1F;
    v23 = 4LL * (v20 >> 5);
    v24 = 4LL * (v18 >> 5);
    while ( 1 )
    {
      v25 = *(_QWORD *)(v19 + 32);
      v26 = v25 + 40LL * (*(_DWORD *)(v19 + 40) & 0xFFFFFF);
      if ( v25 != v26 )
        break;
LABEL_17:
      v19 = *(_QWORD *)(v19 + 8);
      if ( a2 == v19 )
        return v16;
    }
    while ( 1 )
    {
      if ( *(_BYTE *)v25 == 12 )
      {
        v27 = *(_QWORD *)(v25 + 24);
        v28 = *(_DWORD *)(v27 + v23);
        if ( !_bittest(&v28, v21) )
          return 0;
        v29 = *(_DWORD *)(v27 + v24);
        if ( !_bittest(&v29, v22) )
          return 0;
      }
      v25 += 40;
      if ( v26 == v25 )
        goto LABEL_17;
    }
  }
  return v16;
}
