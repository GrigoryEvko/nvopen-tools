// Function: sub_1EDADD0
// Address: 0x1edadd0
//
__int64 __fastcall sub_1EDADD0(__int64 a1, __int64 a2)
{
  unsigned int v3; // eax
  unsigned int v4; // r12d
  unsigned int v5; // eax
  unsigned int v6; // edx
  int v7; // ecx
  __int64 v8; // rax
  __int64 v9; // rsi
  __int64 v10; // r13
  int v11; // edx
  _QWORD *v12; // rax
  __int64 v13; // rcx
  __int64 v14; // rax
  int v15; // ecx
  __int64 v17; // rdx
  unsigned __int64 v18; // r14
  __int64 v19; // r13
  unsigned int v20; // edx
  unsigned __int64 v21; // r13
  __int64 v22; // rax
  __int64 v23; // rdi
  int v24; // ecx
  int v25; // ecx
  __int64 v26; // rdi
  int v27; // [rsp+0h] [rbp-30h] BYREF
  int v28; // [rsp+4h] [rbp-2Ch] BYREF
  unsigned int v29[10]; // [rsp+8h] [rbp-28h] BYREF

  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_WORD *)(a1 + 25) = 0;
  v3 = sub_1ED87E0(*(_QWORD *)a1, a2, &v27, &v28, (int *)v29, (int *)&v29[1]);
  if ( !(_BYTE)v3 )
    return 0;
  v4 = v3;
  v5 = v29[0];
  v6 = v29[1];
  v7 = v27;
  *(_BYTE *)(a1 + 24) = *(_QWORD *)v29 != 0;
  if ( v7 > 0 )
  {
    if ( v28 > 0 )
      return 0;
    v27 = v28;
    v28 = v7;
    *(_QWORD *)v29 = __PAIR64__(v5, v6);
    *(_BYTE *)(a1 + 26) = 1;
  }
  v8 = sub_1E15F70(a2);
  v9 = (unsigned int)v28;
  v10 = *(_QWORD *)(v8 + 40);
  if ( v28 <= 0 )
  {
    v17 = *(_QWORD *)(v10 + 24);
    v18 = *(_QWORD *)(v17 + 16LL * (v27 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL;
    v19 = *(_QWORD *)(v17 + 16LL * (v28 & 0x7FFFFFFF));
    v20 = v29[0];
    v21 = v19 & 0xFFFFFFFFFFFFFFF8LL;
    if ( v29[0] )
    {
      if ( v29[1] )
      {
        if ( v29[0] != v29[1] && v28 == v27 )
          return 0;
        v22 = sub_1F4B080(*(_QWORD *)a1, v18, v29[0], v21, v29[1], (int)a1 + 20, a1 + 16);
        *(_QWORD *)(a1 + 32) = v22;
        if ( !v22 )
          return 0;
        goto LABEL_25;
      }
      v26 = *(_QWORD *)a1;
      *(_DWORD *)(a1 + 16) = v29[0];
      v22 = (*(__int64 (__fastcall **)(__int64, unsigned __int64, unsigned __int64, _QWORD))(*(_QWORD *)v26 + 96LL))(
              v26,
              v18,
              v21,
              v20);
      *(_QWORD *)(a1 + 32) = v22;
    }
    else
    {
      v23 = *(_QWORD *)a1;
      if ( v29[1] )
      {
        *(_DWORD *)(a1 + 20) = v29[1];
        v22 = (*(__int64 (__fastcall **)(__int64, unsigned __int64, unsigned __int64))(*(_QWORD *)v23 + 96LL))(
                v23,
                v21,
                v18);
      }
      else
      {
        v22 = sub_1F4AF90(v23, v21, v18, 255);
      }
      *(_QWORD *)(a1 + 32) = v22;
    }
    if ( !v22 )
      return 0;
LABEL_25:
    v24 = *(_DWORD *)(a1 + 16);
    v11 = v27;
    LODWORD(v9) = v28;
    if ( v24 && !*(_DWORD *)(a1 + 20) )
    {
      *(_BYTE *)(a1 + 26) ^= 1u;
      *(_DWORD *)(a1 + 16) = 0;
      *(_DWORD *)(a1 + 20) = v24;
      v25 = v11;
      v11 = v9;
      LODWORD(v9) = v25;
    }
    *(_BYTE *)(a1 + 25) = v21 != v22 || v18 != v22;
    goto LABEL_10;
  }
  if ( v29[1] )
  {
    v28 = sub_38D6F10(*(_QWORD *)a1 + 8LL, (unsigned int)v28, v29[1]);
    v9 = (unsigned int)v28;
    if ( !v28 )
      return 0;
    v29[1] = 0;
  }
  v11 = v27;
  v12 = (_QWORD *)(*(_QWORD *)(*(_QWORD *)(v10 + 24) + 16LL * (v27 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL);
  if ( v29[0] )
  {
    LODWORD(v9) = sub_38D6F80(*(_QWORD *)a1 + 8LL, v9, v29[0], *v12);
    if ( (_DWORD)v9 )
    {
      v11 = v27;
LABEL_10:
      *(_DWORD *)(a1 + 12) = v11;
      *(_DWORD *)(a1 + 8) = v9;
      return v4;
    }
  }
  else
  {
    v13 = *v12;
    v14 = (unsigned int)v9 >> 3;
    if ( (unsigned int)v14 < *(unsigned __int16 *)(v13 + 22) )
    {
      v15 = *(unsigned __int8 *)(*(_QWORD *)(v13 + 8) + v14);
      if ( _bittest(&v15, v9 & 7) )
        goto LABEL_10;
    }
  }
  return 0;
}
