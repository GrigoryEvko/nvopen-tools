// Function: sub_22C5700
// Address: 0x22c5700
//
__int64 __fastcall sub_22C5700(__int64 a1, _QWORD *a2, _QWORD *a3, int a4)
{
  _QWORD *v6; // r12
  unsigned __int64 *v7; // rbx
  __int64 v8; // r15
  unsigned __int64 *v9; // r15
  __int64 v10; // rax
  __int64 result; // rax
  __int64 v12; // rsi
  unsigned int v13; // eax
  _QWORD *v14; // rdi
  __int64 v15; // rdx
  __int64 v16; // rbx
  int v17; // eax
  __int64 v18; // rax
  __int64 v19; // rdx
  bool v20; // zf
  int v21; // r8d
  _QWORD *v22; // r15
  __int64 v23; // rax
  _QWORD v24[2]; // [rsp+0h] [rbp-70h] BYREF
  __int64 v25; // [rsp+10h] [rbp-60h]
  __int64 v26; // [rsp+20h] [rbp-50h] BYREF
  __int64 v27; // [rsp+28h] [rbp-48h]
  __int64 v28; // [rsp+30h] [rbp-40h]

  v6 = a2;
  v20 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  v26 = 0;
  v27 = 0;
  v28 = -4096;
  if ( v20 )
  {
    v7 = *(unsigned __int64 **)(a1 + 16);
    v8 = 3LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    v7 = (unsigned __int64 *)(a1 + 16);
    v8 = 6;
  }
  v9 = &v7[v8];
  if ( v7 != v9 )
  {
    do
    {
      if ( v7 )
      {
        *v7 = 0;
        v7[1] = 0;
        v10 = v28;
        v20 = v28 == 0;
        v7[2] = v28;
        if ( v10 != -4096 && !v20 && v10 != -8192 )
          sub_BD6050(v7, v26 & 0xFFFFFFFFFFFFFFF8LL);
      }
      v7 += 3;
    }
    while ( v9 != v7 );
    if ( v28 != -4096 && v28 != 0 && v28 != -8192 )
      sub_BD60C0(&v26);
  }
  v24[0] = 0;
  result = -4096;
  v24[1] = 0;
  v25 = -4096;
  v26 = 0;
  v27 = 0;
  v28 = -8192;
  if ( a2 != a3 )
  {
    while ( 1 )
    {
      v16 = v6[2];
      if ( v16 != result && v28 != v16 )
      {
        if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
        {
          v12 = a1 + 16;
          a4 = 1;
        }
        else
        {
          v17 = *(_DWORD *)(a1 + 24);
          v12 = *(_QWORD *)(a1 + 16);
          if ( !v17 )
            BUG();
          a4 = v17 - 1;
        }
        v13 = a4 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
        v14 = (_QWORD *)(v12 + 24LL * v13);
        v15 = v14[2];
        if ( v16 != v15 )
        {
          v21 = 1;
          v22 = 0;
          while ( v15 != -4096 )
          {
            if ( v15 != -8192 || v22 )
              v14 = v22;
            v13 = a4 & (v21 + v13);
            v15 = *(_QWORD *)(v12 + 24LL * v13 + 16);
            if ( v16 == v15 )
              goto LABEL_16;
            v22 = v14;
            ++v21;
            v14 = (_QWORD *)(v12 + 24LL * v13);
          }
          if ( v22 )
          {
            v23 = v22[2];
          }
          else
          {
            v23 = v14[2];
            v22 = v14;
          }
          if ( v16 != v23 )
          {
            if ( v23 != -4096 && v23 != 0 && v23 != -8192 )
              sub_BD60C0(v22);
            v22[2] = v16;
            if ( v16 != 0 && v16 != -4096 && v16 != -8192 )
              sub_BD73F0((__int64)v22);
          }
        }
LABEL_16:
        *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
        v16 = v6[2];
      }
      if ( v16 != -4096 && v16 != 0 && v16 != -8192 )
        sub_BD60C0(v6);
      v6 += 3;
      if ( a3 == v6 )
        break;
      result = v25;
    }
    LODWORD(v18) = v28;
    if ( v28 != -4096 && v28 != 0 && v28 != -8192 )
    {
      v18 = sub_BD60C0(&v26);
      v19 = v25;
      LOBYTE(v18) = v25 != -4096;
      v20 = v25 == 0;
    }
    else
    {
      v19 = v25;
      LOBYTE(v18) = v25 != 0;
      v20 = v25 == -4096;
    }
    LOBYTE(a4) = !v20;
    LOBYTE(v19) = v19 != -8192;
    result = (unsigned int)v19 & a4 & (unsigned int)v18;
    if ( (_BYTE)result )
      return sub_BD60C0(v24);
  }
  return result;
}
