// Function: sub_1E6A350
// Address: 0x1e6a350
//
__int64 __fastcall sub_1E6A350(_QWORD *a1, unsigned int a2, char a3)
{
  unsigned int v3; // ecx
  __int64 (*v6)(); // rax
  __int64 v7; // rax
  _QWORD *v8; // rdx
  __int64 v9; // rdi
  __int64 v10; // r13
  unsigned int v11; // eax
  __int16 v12; // si
  _WORD *v13; // rax
  _WORD *v14; // rdi
  unsigned __int16 v15; // cx
  __int64 v16; // rsi
  __int64 v17; // r9
  __int64 v18; // rax
  __int64 v19; // r14
  __int64 v20; // r15
  __int16 v21; // ax
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rdi
  __int16 v28; // ax
  unsigned int v29; // [rsp+10h] [rbp-70h] BYREF
  __int64 v30; // [rsp+18h] [rbp-68h]
  char v31; // [rsp+20h] [rbp-60h]
  unsigned __int16 v32; // [rsp+28h] [rbp-58h]
  _WORD *v33; // [rsp+30h] [rbp-50h]
  int v34; // [rsp+38h] [rbp-48h]
  unsigned __int16 v35; // [rsp+40h] [rbp-40h]
  __int64 v36; // [rsp+48h] [rbp-38h]

  v3 = a2;
  if ( (*(_QWORD *)(a1[35] + 8LL * (a2 >> 6)) & (1LL << a2)) == 0 )
  {
    v6 = *(__int64 (**)())(**(_QWORD **)(*a1 + 16LL) + 112LL);
    if ( v6 == sub_1D00B10 || (v7 = v6(), v3 = a2, (v8 = (_QWORD *)v7) == 0) )
    {
      v29 = v3;
      v30 = 0;
      v31 = 1;
      v32 = 0;
      v33 = 0;
      v34 = 0;
      v35 = 0;
      v36 = 0;
      BUG();
    }
    v31 = 1;
    v32 = 0;
    v33 = 0;
    v36 = 0;
    v9 = *(_QWORD *)(v7 + 56);
    v30 = v7 + 8;
    v34 = 0;
    v35 = 0;
    v10 = *(_QWORD *)(v7 + 8);
    v29 = a2;
    v11 = *(_DWORD *)(v10 + 24LL * a2 + 16);
    v12 = a2 * (v11 & 0xF);
    v13 = (_WORD *)(v9 + 2LL * (v11 >> 4));
    v14 = v13 + 1;
    v32 = *v13 + v12;
    v33 = v13 + 1;
    while ( 1 )
    {
      if ( !v14 )
        return 0;
      v34 = *(_DWORD *)(v8[6] + 4LL * v32);
      v15 = v34;
      if ( (_WORD)v34 )
        break;
LABEL_34:
      v33 = ++v14;
      v28 = *(v14 - 1);
      v32 += v28;
      if ( !v28 )
        return 0;
    }
    while ( 1 )
    {
      v16 = v15;
      v17 = *(unsigned int *)(v8[1] + 24LL * v15 + 8);
      v18 = v8[7];
      v35 = v15;
      v36 = v18 + 2 * v17;
      if ( v36 )
        break;
      v15 = HIWORD(v34);
      v34 = HIWORD(v34);
      if ( !v15 )
        goto LABEL_34;
    }
    while ( 1 )
    {
      v19 = *(_QWORD *)(a1[34] + 8 * v16);
      if ( v19 )
      {
        if ( (*(_BYTE *)(v19 + 3) & 0x10) != 0 )
          break;
        v19 = *(_QWORD *)(v19 + 32);
        if ( v19 )
        {
          if ( (*(_BYTE *)(v19 + 3) & 0x10) != 0 )
            break;
        }
      }
LABEL_19:
      sub_1E1D5E0((__int64)&v29);
      if ( !v33 )
        return 0;
      v16 = v35;
    }
    if ( !a3 )
    {
      while ( 1 )
      {
        v20 = *(_QWORD *)(v19 + 16);
        v21 = *(_WORD *)(v20 + 46);
        if ( (v21 & 4) == 0 && (v21 & 8) != 0 )
          LOBYTE(v22) = sub_1E15D00(*(_QWORD *)(v19 + 16), 0x10u, 1);
        else
          v22 = (*(_QWORD *)(*(_QWORD *)(v20 + 16) + 8LL) >> 4) & 1LL;
        if ( !(_BYTE)v22 )
          break;
        v23 = *(_QWORD *)(v20 + 24);
        if ( *(_QWORD *)(v23 + 96) != *(_QWORD *)(v23 + 88) )
          break;
        if ( (unsigned __int8)sub_1560180(**(_QWORD **)(v23 + 56) + 112LL, 56) )
          break;
        v25 = *(_QWORD *)(v20 + 32);
        v26 = v25 + 40LL * *(unsigned int *)(v20 + 40);
        if ( v25 == v26 )
          break;
        while ( 1 )
        {
          if ( *(_BYTE *)v25 == 10 )
          {
            v27 = *(_QWORD *)(v25 + 24);
            if ( !*(_BYTE *)(v27 + 16) )
              break;
          }
          v25 += 40;
          if ( v26 == v25 )
            return 1;
        }
        if ( !(unsigned __int8)sub_1560180(v27 + 112, 29) || !(unsigned __int8)sub_1560180(v27 + 112, 30) )
          break;
        v19 = *(_QWORD *)(v19 + 32);
        if ( !v19 || (*(_BYTE *)(v19 + 3) & 0x10) == 0 )
          goto LABEL_19;
      }
    }
  }
  return 1;
}
