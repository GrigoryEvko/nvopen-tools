// Function: sub_257C940
// Address: 0x257c940
//
__int64 __fastcall sub_257C940(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rax
  unsigned int v9; // eax
  __int64 v10; // r14
  int v11; // r14d
  __int64 v12; // rbx
  __int64 v13; // rax
  unsigned int v14; // edx
  unsigned int v15; // eax
  __int64 v16; // r9
  __int64 v17; // rdx
  unsigned __int64 v18; // rax
  __int64 v19; // r12
  int v20; // ecx
  unsigned __int64 v21; // r8
  unsigned __int64 v22; // rsi
  __int64 v23; // rax
  int v25; // r10d
  int v26; // r9d
  unsigned int v27; // eax
  __int64 v28; // rsi
  int v29; // eax
  __int64 v30; // rcx
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // rdx
  __int64 v36; // rsi
  __int64 v37; // rdx
  __int64 v38; // rdi
  __int64 (__fastcall *v39)(__int64); // rax
  __int64 v40; // rdi
  int v41; // eax
  __int64 v42; // [rsp+0h] [rbp-C0h]
  int v43; // [rsp+0h] [rbp-C0h]
  __int64 v44; // [rsp+8h] [rbp-B8h]
  __int64 v45; // [rsp+8h] [rbp-B8h]
  unsigned int v49; // [rsp+2Ch] [rbp-94h]
  void *v50; // [rsp+30h] [rbp-90h]
  __int64 v51; // [rsp+38h] [rbp-88h]
  unsigned __int64 v52; // [rsp+40h] [rbp-80h]
  __int64 v53; // [rsp+48h] [rbp-78h]
  char v54[8]; // [rsp+50h] [rbp-70h] BYREF
  __int64 v55; // [rsp+58h] [rbp-68h]
  unsigned int v56; // [rsp+68h] [rbp-58h]
  __int64 v57; // [rsp+78h] [rbp-48h]
  __int64 v58; // [rsp+80h] [rbp-40h]
  __int64 v59; // [rsp+88h] [rbp-38h]

  v8 = sub_2568740(a3, a4);
  sub_254C700((__int64)v54, v8);
  sub_C7D6A0(0, 0, 8);
  v9 = *(_DWORD *)(a3 + 224);
  v50 = 0;
  v49 = v9;
  if ( v9 )
  {
    v10 = 8LL * v9;
    v50 = (void *)sub_C7D670(v10, 8);
    memcpy(v50, *(const void **)(a3 + 208), v10);
  }
  v11 = 0;
  v12 = *(_QWORD *)(a3 + 240);
  v53 = *(_QWORD *)(a3 + 248);
  v51 = *(_QWORD *)(a3 + 256);
  v13 = 0;
  if ( *(_DWORD *)(a5 + 40) )
  {
    while ( 1 )
    {
      v18 = *(_QWORD *)(*(_QWORD *)(a5 + 32) + 8 * v13);
      v19 = *(_QWORD *)(v18 + 24);
      v52 = v18;
      v20 = *(unsigned __int8 *)v19;
      if ( (unsigned __int8)v20 <= 0x1Cu )
        goto LABEL_9;
      v21 = v19 & 0xFFFFFFFFFFFFFFFBLL;
      v22 = v19 | 4;
      if ( !v56 )
        goto LABEL_12;
      v14 = v56 - 1;
      v15 = (v56 - 1) & (v22 ^ (v22 >> 9));
      v16 = *(_QWORD *)(v55 + 8LL * v15);
      if ( v16 != v22 )
      {
        v25 = 1;
        while ( v16 != -4 )
        {
          v15 = v14 & (v25 + v15);
          v16 = *(_QWORD *)(v55 + 8LL * v15);
          if ( v22 == v16 )
            goto LABEL_6;
          ++v25;
        }
        v26 = 1;
        v27 = v14 & (v21 ^ (v21 >> 9));
        v28 = *(_QWORD *)(v55 + 8LL * v27);
        if ( v21 != v28 )
          break;
      }
LABEL_6:
      if ( (unsigned __int8)(v20 - 34) <= 0x33u )
      {
        v17 = 0x8000000000041LL;
        if ( _bittest64(&v17, (unsigned int)(v20 - 34)) )
        {
          if ( v52 >= v19 - 32 * (unsigned __int64)(*(_DWORD *)(v19 + 4) & 0x7FFFFFF) )
          {
            v29 = v20 - 29;
            if ( v20 == 40 )
            {
              v30 = -32 - 32LL * (unsigned int)sub_B491D0(v19);
            }
            else
            {
              v30 = -32;
              if ( v29 != 56 )
              {
                if ( v29 != 5 )
                  BUG();
                v30 = -96;
              }
            }
            if ( *(char *)(v19 + 7) < 0 )
            {
              v42 = v30;
              v31 = sub_BD2BC0(v19);
              v30 = v42;
              v44 = v32 + v31;
              if ( *(char *)(v19 + 7) >= 0 )
              {
                if ( (unsigned int)(v44 >> 4) )
LABEL_47:
                  BUG();
              }
              else
              {
                v33 = sub_BD2BC0(v19);
                v30 = v42;
                if ( (unsigned int)((v44 - v33) >> 4) )
                {
                  v45 = v42;
                  if ( *(char *)(v19 + 7) >= 0 )
                    goto LABEL_47;
                  v43 = *(_DWORD *)(sub_BD2BC0(v19) + 8);
                  if ( *(char *)(v19 + 7) >= 0 )
                    BUG();
                  v34 = sub_BD2BC0(v19);
                  v30 = v45 - 32LL * (unsigned int)(*(_DWORD *)(v34 + v35 - 4) - v43);
                }
              }
            }
            if ( v52 < v19 + v30 )
            {
              v36 = sub_254C9B0(v19, (__int64)(v52 - (v19 - 32LL * (*(_DWORD *)(v19 + 4) & 0x7FFFFFF))) >> 5);
              v38 = sub_257C550(a2, v36, v37, a1, 2, 0, 1);
              if ( v38 )
              {
                v39 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v38 + 48LL);
                if ( v39 == sub_2534FF0 )
                  v40 = v38 + 88;
                else
                  v40 = ((__int64 (__fastcall *)(__int64, __int64))v39)(v38, v36);
                v41 = *(_DWORD *)(v40 + 8);
                *(_DWORD *)(a6 + 12) |= v41;
                *(_DWORD *)(a6 + 8) |= v41;
              }
            }
          }
        }
      }
LABEL_9:
      v13 = (unsigned int)(v11 + 1);
      v11 = v13;
      if ( *(_DWORD *)(a5 + 40) <= (unsigned int)v13 )
        goto LABEL_19;
    }
    while ( v28 != -4 )
    {
      v27 = v14 & (v26 + v27);
      v28 = *(_QWORD *)(v55 + 8LL * v27);
      if ( v21 == v28 )
        goto LABEL_6;
      ++v26;
    }
LABEL_12:
    v23 = v57;
    do
    {
      if ( v12 == v23 && v53 == v58 && v51 == v59 )
        goto LABEL_9;
      v23 = sub_3106C80(v54);
      v57 = v23;
    }
    while ( v19 != v23 );
    v20 = *(unsigned __int8 *)v19;
    goto LABEL_6;
  }
LABEL_19:
  sub_C7D6A0((__int64)v50, 8LL * v49, 8);
  return sub_C7D6A0(v55, 8LL * v56, 8);
}
