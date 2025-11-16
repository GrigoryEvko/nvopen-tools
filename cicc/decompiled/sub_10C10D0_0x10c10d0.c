// Function: sub_10C10D0
// Address: 0x10c10d0
//
unsigned __int8 *__fastcall sub_10C10D0(char *a1, __int64 *a2)
{
  unsigned __int8 v2; // al
  __int64 v3; // r13
  __int64 v4; // r14
  __int64 v5; // r15
  __int64 v6; // rax
  unsigned __int8 *v7; // r12
  __int64 v9; // r15
  int v10; // edx
  unsigned int v11; // eax
  int v12; // ecx
  unsigned int v13; // edx
  unsigned int v14; // eax
  __int64 v15; // rcx
  unsigned __int64 v16; // rcx
  unsigned __int64 v17; // rax
  __int64 v18; // r12
  __int64 v19; // rax
  unsigned int v20; // edi
  unsigned __int64 v21; // rax
  __int64 v22; // rdx
  _BYTE *v23; // rax
  unsigned int v25; // ecx
  __int64 v27; // rdx
  int v28; // r15d
  __int64 v29; // r15
  __int64 v30; // rbx
  __int64 v31; // rdx
  unsigned int v32; // esi
  __int64 v33; // [rsp+0h] [rbp-C0h]
  __int64 v34; // [rsp+8h] [rbp-B8h]
  int v35; // [rsp+10h] [rbp-B0h]
  unsigned int v36; // [rsp+10h] [rbp-B0h]
  unsigned int v37; // [rsp+10h] [rbp-B0h]
  unsigned __int8 v38; // [rsp+1Ch] [rbp-A4h]
  unsigned int v39; // [rsp+1Ch] [rbp-A4h]
  __int64 v40; // [rsp+20h] [rbp-A0h] BYREF
  __int64 *v41; // [rsp+28h] [rbp-98h] BYREF
  char v42[32]; // [rsp+30h] [rbp-90h] BYREF
  __int16 v43; // [rsp+50h] [rbp-70h]
  __int64 *v44; // [rsp+60h] [rbp-60h] BYREF
  __int64 *v45; // [rsp+68h] [rbp-58h] BYREF
  char v46; // [rsp+70h] [rbp-50h]
  __int16 v47; // [rsp+80h] [rbp-40h]

  v2 = *a1;
  v3 = *((_QWORD *)a1 - 8);
  v46 = 0;
  v4 = *((_QWORD *)a1 + 1);
  v5 = *((_QWORD *)a1 - 4);
  v38 = v2;
  v44 = &v40;
  v45 = (__int64 *)&v41;
  v6 = *(_QWORD *)(v3 + 16);
  if ( !v6 )
    return 0;
  v7 = *(unsigned __int8 **)(v6 + 8);
  if ( v7 )
    return 0;
  if ( *(_BYTE *)v3 != 42 )
    return 0;
  if ( !*(_QWORD *)(v3 - 64) )
    return 0;
  v40 = *(_QWORD *)(v3 - 64);
  if ( !(unsigned __int8)sub_991580((__int64)&v45, *(_QWORD *)(v3 - 32)) )
    return 0;
  if ( *(_BYTE *)v5 != 17 )
  {
    v22 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v5 + 8) + 8LL) - 17;
    if ( (unsigned int)v22 <= 1 && *(_BYTE *)v5 <= 0x15u )
    {
      v23 = sub_AD7630(v5, 0, v22);
      if ( v23 )
      {
        v9 = (__int64)(v23 + 24);
        if ( *v23 == 17 )
          goto LABEL_10;
      }
    }
    return 0;
  }
  v9 = v5 + 24;
LABEL_10:
  v10 = sub_BCB060(v4);
  v11 = *((_DWORD *)v41 + 2);
  if ( v11 <= 0x40 )
  {
    _RDI = *v41;
    v25 = 64;
    __asm { tzcnt   r9, rdi }
    if ( *v41 )
      v25 = _R9;
    if ( v11 > v25 )
      v11 = v25;
  }
  else
  {
    v35 = v10;
    v11 = sub_C44590((__int64)v41);
    v10 = v35;
  }
  v12 = v38;
  v13 = v10 - v11;
  v39 = v38 - 29;
  if ( v12 == 57 )
  {
    v14 = *(_DWORD *)(v9 + 8);
    if ( v14 > 0x40 )
    {
      v36 = v13;
      v14 = sub_C44500(v9);
      v13 = v36;
    }
    else if ( v14 )
    {
      v15 = *(_QWORD *)v9 << (64 - (unsigned __int8)v14);
      v14 = 64;
      v16 = ~v15;
      if ( v16 )
      {
        _BitScanReverse64(&v17, v16);
        v14 = v17 ^ 0x3F;
      }
    }
  }
  else
  {
    if ( (unsigned __int8)(v12 - 58) > 1u )
      BUG();
    v14 = *(_DWORD *)(v9 + 8);
    if ( v14 > 0x40 )
    {
      v37 = v13;
      v14 = sub_C444A0(v9);
      v13 = v37;
    }
    else
    {
      v20 = v14 - 64;
      if ( *(_QWORD *)v9 )
      {
        _BitScanReverse64(&v21, *(_QWORD *)v9);
        v14 = v20 + (v21 ^ 0x3F);
      }
    }
  }
  if ( v13 <= v14 )
  {
    v43 = 257;
    v33 = sub_AD8D80(v4, v9);
    v34 = v40;
    v18 = (*(__int64 (__fastcall **)(__int64, _QWORD, __int64, __int64))(*(_QWORD *)a2[10] + 16LL))(
            a2[10],
            v39,
            v40,
            v33);
    if ( !v18 )
    {
      v47 = 257;
      v18 = sub_B504D0(v39, v34, v33, (__int64)&v44, 0, 0);
      if ( (unsigned __int8)sub_920620(v18) )
      {
        v27 = a2[12];
        v28 = *((_DWORD *)a2 + 26);
        if ( v27 )
          sub_B99FD0(v18, 3u, v27);
        sub_B45150(v18, v28);
      }
      (*(void (__fastcall **)(__int64, __int64, char *, __int64, __int64))(*(_QWORD *)a2[11] + 16LL))(
        a2[11],
        v18,
        v42,
        a2[7],
        a2[8]);
      v29 = *a2;
      v30 = *a2 + 16LL * *((unsigned int *)a2 + 2);
      if ( *a2 != v30 )
      {
        do
        {
          v31 = *(_QWORD *)(v29 + 8);
          v32 = *(_DWORD *)v29;
          v29 += 16;
          sub_B99FD0(v18, v32, v31);
        }
        while ( v30 != v29 );
      }
    }
    v47 = 257;
    v19 = sub_AD8D80(v4, (__int64)v41);
    v7 = (unsigned __int8 *)sub_B504D0(13, v18, v19, (__int64)&v44, 0, 0);
    sub_B45260(v7, v3, 1);
  }
  return v7;
}
