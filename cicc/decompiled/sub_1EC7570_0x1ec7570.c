// Function: sub_1EC7570
// Address: 0x1ec7570
//
__int64 __fastcall sub_1EC7570(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v7; // r9d
  int v8; // r11d
  __int64 v9; // r10
  __int64 v10; // rax
  unsigned __int16 *v11; // rdi
  unsigned int v12; // r14d
  unsigned __int16 *v13; // rsi
  __int64 v14; // rcx
  __int64 v16; // rdx
  unsigned __int16 *v17; // rax
  __int64 v18; // rax
  unsigned int v19; // ebx
  unsigned __int16 *v20; // rdi
  __int64 v21; // rax
  __int64 v22; // rcx
  __int64 v23; // rdi
  int v24; // eax
  __int64 v25; // rdx
  __int64 v26; // rcx
  _QWORD *v27; // r9
  float *v28; // r8
  unsigned int v29; // r8d
  unsigned int v30; // eax
  __int64 v32; // [rsp+10h] [rbp-40h] BYREF
  float v33[14]; // [rsp+18h] [rbp-38h] BYREF

  v7 = -*(_DWORD *)(a3 + 8);
  for ( *(_DWORD *)(a3 + 64) = v7; ; v7 = *(_DWORD *)(a3 + 64) )
  {
    if ( v7 >= 0 )
    {
      if ( !*(_BYTE *)(a3 + 68) )
      {
        v8 = *(_DWORD *)(a3 + 56);
        v9 = 2LL * v7;
        while ( v7 < v8 )
        {
          v10 = *(_QWORD *)(a3 + 48);
          v11 = *(unsigned __int16 **)a3;
          *(_DWORD *)(a3 + 64) = v7 + 1;
          v12 = *(unsigned __int16 *)(v10 + v9);
          v13 = &v11[*(unsigned int *)(a3 + 8)];
          LODWORD(v33[0]) = v12;
          if ( v13 == sub_1EBB4B0(v11, (__int64)v13, (int *)v33) )
            goto LABEL_10;
        }
      }
      return 0;
    }
    v16 = *(unsigned int *)(a3 + 8);
    v14 = (unsigned int)(v7 + 1);
    v17 = *(unsigned __int16 **)a3;
    *(_DWORD *)(a3 + 64) = v14;
    v12 = v17[v16 + v7];
LABEL_10:
    if ( !v12 )
      return 0;
    if ( !(unsigned int)sub_21038C0(a1[34], a2, v12, v14) )
      break;
  }
  if ( *(int *)(a3 + 64) > 0 )
  {
    v18 = *(_QWORD *)(a1[31] + 208LL) + 40LL * (*(_DWORD *)(a2 + 112) & 0x7FFFFFFF);
    if ( *(_DWORD *)(v18 + 16) )
    {
      if ( !*(_DWORD *)v18 )
      {
        v19 = **(_DWORD **)(v18 + 8);
        if ( v19 )
        {
          v20 = *(unsigned __int16 **)a3;
          v21 = *(unsigned int *)(a3 + 8);
          LODWORD(v33[0]) = v19;
          if ( &v20[v21] != sub_1EBB4B0(v20, (__int64)&v20[v21], (int *)v33) )
          {
            v23 = a1[34];
            *(_QWORD *)v33 = 1;
            v24 = sub_21038C0(v23, a2, v19, v22);
            v28 = v33;
            if ( v24 <= 1 && (unsigned __int8)sub_1EBD330((__int64)a1, a2, v19, 1u, v33) )
            {
              v12 = v19;
              sub_1EC63D0((__int64)a1, a2, v19, a4, v28, (__int64)v27);
              return v12;
            }
            v32 = a2;
            sub_1EC7290((__int64)(a1 + 3427), &v32, v25, v26, v28, v27);
          }
        }
      }
    }
    v29 = *(_DWORD *)(*(_QWORD *)(a1[87] + 232LL) + 8LL * v12);
    if ( v29 )
    {
      v30 = sub_1EC6BB0((__int64)a1, a2, a3, a4, v29);
      if ( v30 )
        return v30;
    }
  }
  return v12;
}
