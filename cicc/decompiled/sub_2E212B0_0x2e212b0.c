// Function: sub_2E212B0
// Address: 0x2e212b0
//
__int64 __fastcall sub_2E212B0(_QWORD *a1, __int64 a2, unsigned int a3)
{
  __int64 result; // rax
  int v5; // eax
  _QWORD *v6; // rcx
  _QWORD *v7; // r12
  __int64 v8; // rsi
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rdx
  unsigned int v12; // r9d
  _QWORD *v13; // r15
  __int16 *v14; // r14
  __int64 v15; // rsi
  int v16; // eax
  unsigned int v17; // eax
  unsigned int v18; // r12d
  __int16 *i; // r14
  _QWORD *v20; // r15
  __int64 v21; // rsi
  int v22; // edx
  _QWORD *v23; // rax
  _QWORD *v24; // r10
  unsigned int v25; // r9d
  __int64 v26; // rdx
  _QWORD *v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // [rsp-88h] [rbp-88h]
  unsigned int v32; // [rsp-7Ch] [rbp-7Ch]
  _QWORD *v33; // [rsp-78h] [rbp-78h]
  unsigned int v34; // [rsp-78h] [rbp-78h]
  unsigned int v35; // [rsp-70h] [rbp-70h]
  char v36; // [rsp-70h] [rbp-70h]
  char v37; // [rsp-70h] [rbp-70h]
  _QWORD *v38; // [rsp-68h] [rbp-68h] BYREF
  unsigned int v39; // [rsp-60h] [rbp-60h]
  int v40; // [rsp-5Ch] [rbp-5Ch]
  __int64 v41; // [rsp-58h] [rbp-58h]
  __int16 v42; // [rsp-50h] [rbp-50h]
  char v43; // [rsp-4Eh] [rbp-4Eh]
  __int64 v44; // [rsp-48h] [rbp-48h]

  result = 0;
  if ( !*(_DWORD *)(a2 + 8) )
    return result;
  v5 = *(_DWORD *)(a2 + 112);
  v6 = (_QWORD *)*a1;
  v39 = a3;
  v7 = *(_QWORD **)(a2 + 104);
  v43 = 0;
  v40 = v5;
  v8 = v6[7];
  v42 = 0;
  v38 = v6;
  v9 = 24LL * a3;
  v41 = 0;
  v44 = 0;
  if ( !v7 )
  {
    v17 = *(_DWORD *)(v6[1] + v9 + 16);
    v18 = v17 & 0xFFF;
    for ( i = (__int16 *)(v8 + 2LL * (v17 >> 12)); i; ++i )
    {
      v20 = (_QWORD *)a1[1];
      v21 = *(_QWORD *)(v20[53] + 8LL * v18);
      if ( !v21 )
      {
        v37 = qword_501EA48[8];
        v27 = (_QWORD *)sub_22077B0(0x68u);
        v28 = v18;
        v21 = (__int64)v27;
        if ( v27 )
        {
          *v27 = v27 + 2;
          v27[1] = 0x200000000LL;
          v27[8] = v27 + 10;
          v27[9] = 0x200000000LL;
          if ( v37 )
          {
            v30 = sub_22077B0(0x30u);
            v28 = v18;
            if ( v30 )
            {
              *(_DWORD *)(v30 + 8) = 0;
              *(_QWORD *)(v30 + 16) = 0;
              *(_QWORD *)(v30 + 24) = v30 + 8;
              *(_QWORD *)(v30 + 32) = v30 + 8;
              *(_QWORD *)(v30 + 40) = 0;
            }
            *(_QWORD *)(v21 + 96) = v30;
          }
          else
          {
            v27[12] = 0;
          }
        }
        *(_QWORD *)(v20[53] + 8 * v28) = v21;
        sub_2E11710(v20, v21, v18);
      }
      result = sub_2E09F20(a2, v21, (__int64)&v38);
      if ( (_BYTE)result )
        return result;
      v22 = *i;
      v18 += v22;
      if ( !(_WORD)v22 )
        return 0;
    }
    return 0;
  }
  v10 = v6[1] + v9;
  v11 = *(_DWORD *)(v10 + 16) >> 12;
  v12 = *(_DWORD *)(v10 + 16) & 0xFFF;
  v13 = (_QWORD *)(v6[8] + 16LL * *(unsigned __int16 *)(v10 + 20));
  if ( !(v8 + 2 * v11) )
    return 0;
  v14 = (__int16 *)(v8 + 2 * v11);
  while ( 1 )
  {
    if ( v7 )
    {
      while ( (v7[14] & *v13) == 0 && (v13[1] & v7[15]) == 0 )
      {
        v7 = (_QWORD *)v7[13];
        if ( !v7 )
          goto LABEL_12;
      }
      v15 = *(_QWORD *)(*(_QWORD *)(a1[1] + 424LL) + 8LL * v12);
      if ( !v15 )
      {
        v31 = v12;
        v32 = v12;
        v33 = (_QWORD *)a1[1];
        v36 = qword_501EA48[8];
        v23 = (_QWORD *)sub_22077B0(0x68u);
        v24 = v33;
        v25 = v32;
        v26 = v31;
        v15 = (__int64)v23;
        if ( v23 )
        {
          *v23 = v23 + 2;
          v23[1] = 0x200000000LL;
          v23[8] = v23 + 10;
          v23[9] = 0x200000000LL;
          if ( v36 )
          {
            v29 = sub_22077B0(0x30u);
            v24 = v33;
            v25 = v32;
            v26 = v31;
            if ( v29 )
            {
              *(_DWORD *)(v29 + 8) = 0;
              *(_QWORD *)(v29 + 16) = 0;
              *(_QWORD *)(v29 + 24) = v29 + 8;
              *(_QWORD *)(v29 + 32) = v29 + 8;
              *(_QWORD *)(v29 + 40) = 0;
            }
            *(_QWORD *)(v15 + 96) = v29;
          }
          else
          {
            v23[12] = 0;
          }
        }
        v34 = v25;
        *(_QWORD *)(v24[53] + 8 * v26) = v15;
        sub_2E11710(v24, v15, v25);
        v12 = v34;
      }
      v35 = v12;
      result = sub_2E09F20((__int64)v7, v15, (__int64)&v38);
      v12 = v35;
      if ( (_BYTE)result )
        return result;
    }
LABEL_12:
    v16 = *v14;
    v13 += 2;
    ++v14;
    if ( !(_WORD)v16 )
      return 0;
    v7 = *(_QWORD **)(a2 + 104);
    v12 += v16;
  }
}
