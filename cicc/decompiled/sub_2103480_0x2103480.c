// Function: sub_2103480
// Address: 0x2103480
//
__int64 __fastcall sub_2103480(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 result; // rax
  _QWORD *v7; // rax
  int v8; // edx
  bool v9; // zf
  __int64 v10; // rdi
  _WORD *v11; // r8
  unsigned __int16 *v12; // rdx
  int v13; // r9d
  _DWORD *v14; // r14
  unsigned __int16 *v15; // r15
  __int64 v16; // r12
  __int64 v17; // rsi
  int v18; // eax
  _QWORD *v19; // rax
  int v20; // r9d
  _QWORD *v21; // r10
  __int64 v22; // rdx
  unsigned int v23; // esi
  _WORD *v24; // rax
  __int16 *v25; // r15
  unsigned __int16 v26; // r12
  _QWORD *v27; // r14
  __int64 v28; // rsi
  __int16 v29; // ax
  _QWORD *v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // [rsp-88h] [rbp-88h]
  _QWORD *v35; // [rsp-80h] [rbp-80h]
  int v36; // [rsp-78h] [rbp-78h]
  int v37; // [rsp-78h] [rbp-78h]
  int v38; // [rsp-70h] [rbp-70h]
  char v39; // [rsp-70h] [rbp-70h]
  char v40; // [rsp-70h] [rbp-70h]
  _QWORD *v41; // [rsp-68h] [rbp-68h] BYREF
  unsigned int v42; // [rsp-60h] [rbp-60h]
  int v43; // [rsp-5Ch] [rbp-5Ch]
  __int64 v44; // [rsp-58h] [rbp-58h]
  __int16 v45; // [rsp-50h] [rbp-50h]
  char v46; // [rsp-4Eh] [rbp-4Eh]
  __int64 v47; // [rsp-48h] [rbp-48h]

  result = 0;
  if ( !*(_DWORD *)(a2 + 8) )
    return result;
  v7 = *(_QWORD **)(a1 + 232);
  v8 = *(_DWORD *)(a2 + 112);
  v9 = *(_QWORD *)(a2 + 104) == 0;
  v42 = a3;
  v41 = v7;
  v43 = v8;
  v44 = 0;
  v45 = 0;
  v46 = 0;
  v47 = 0;
  if ( !v9 )
  {
    if ( !v7 )
      BUG();
    v10 = v7[1] + 24LL * a3;
    v13 = a3 * (*(_DWORD *)(v10 + 16) & 0xF);
    v11 = (_WORD *)(v7[7] + 2LL * (*(_DWORD *)(v10 + 16) >> 4));
    v12 = v11 + 1;
    LOWORD(v13) = *v11 + v13;
    v14 = (_DWORD *)(v7[8] + 4LL * *(unsigned __int16 *)(v10 + 20));
LABEL_5:
    v15 = v12;
    if ( v12 )
    {
      while ( 1 )
      {
        v16 = *(_QWORD *)(a2 + 104);
        if ( v16 )
        {
          while ( (*(_DWORD *)(v16 + 112) & *v14) == 0 )
          {
            v16 = *(_QWORD *)(v16 + 104);
            if ( !v16 )
              goto LABEL_12;
          }
          v17 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 240) + 672LL) + 8LL * (unsigned __int16)v13);
          if ( !v17 )
          {
            v34 = (unsigned __int16)v13;
            v35 = *(_QWORD **)(a1 + 240);
            v36 = v13;
            v39 = qword_4FC4440[20];
            v19 = (_QWORD *)sub_22077B0(104);
            v20 = v36;
            v21 = v35;
            v22 = v34;
            v17 = (__int64)v19;
            if ( v19 )
            {
              *v19 = v19 + 2;
              v19[1] = 0x200000000LL;
              v19[8] = v19 + 10;
              v19[9] = 0x200000000LL;
              if ( v39 )
              {
                v33 = sub_22077B0(48);
                v20 = v36;
                v21 = v35;
                v22 = v34;
                if ( v33 )
                {
                  *(_DWORD *)(v33 + 8) = 0;
                  *(_QWORD *)(v33 + 16) = 0;
                  *(_QWORD *)(v33 + 24) = v33 + 8;
                  *(_QWORD *)(v33 + 32) = v33 + 8;
                  *(_QWORD *)(v33 + 40) = 0;
                }
                *(_QWORD *)(v17 + 96) = v33;
              }
              else
              {
                v19[12] = 0;
              }
            }
            v37 = v20;
            *(_QWORD *)(v21[84] + 8 * v22) = v17;
            sub_1DBA8F0(v21, v17, (unsigned __int16)v20);
            v13 = v37;
          }
          v38 = v13;
          result = sub_1DB3E90(v16, v17, (__int64)&v41);
          v13 = v38;
          if ( (_BYTE)result )
            return result;
        }
LABEL_12:
        v18 = *v15;
        ++v14;
        ++v15;
        v12 = 0;
        if ( !(_WORD)v18 )
          goto LABEL_5;
        v13 += v18;
        if ( !v15 )
          return 0;
      }
    }
    return 0;
  }
  if ( !v7 )
    BUG();
  v23 = *(_DWORD *)(v7[1] + 24LL * a3 + 16);
  v24 = (_WORD *)(v7[7] + 2LL * (v23 >> 4));
  v25 = v24 + 1;
  v26 = *v24 + a3 * (v23 & 0xF);
LABEL_22:
  if ( !v25 )
    return 0;
  while ( 1 )
  {
    v27 = *(_QWORD **)(a1 + 240);
    v28 = *(_QWORD *)(v27[84] + 8LL * v26);
    if ( !v28 )
    {
      v40 = qword_4FC4440[20];
      v30 = (_QWORD *)sub_22077B0(104);
      v31 = v26;
      v28 = (__int64)v30;
      if ( v30 )
      {
        *v30 = v30 + 2;
        v30[1] = 0x200000000LL;
        v30[8] = v30 + 10;
        v30[9] = 0x200000000LL;
        if ( v40 )
        {
          v32 = sub_22077B0(48);
          v31 = v26;
          if ( v32 )
          {
            *(_DWORD *)(v32 + 8) = 0;
            *(_QWORD *)(v32 + 16) = 0;
            *(_QWORD *)(v32 + 24) = v32 + 8;
            *(_QWORD *)(v32 + 32) = v32 + 8;
            *(_QWORD *)(v32 + 40) = 0;
          }
          *(_QWORD *)(v28 + 96) = v32;
        }
        else
        {
          v30[12] = 0;
        }
      }
      *(_QWORD *)(v27[84] + 8 * v31) = v28;
      sub_1DBA8F0(v27, v28, v26);
    }
    result = sub_1DB3E90(a2, v28, (__int64)&v41);
    if ( (_BYTE)result )
      return result;
    v29 = *v25++;
    if ( !v29 )
    {
      v25 = 0;
      goto LABEL_22;
    }
    v26 += v29;
    if ( !v25 )
      return 0;
  }
}
