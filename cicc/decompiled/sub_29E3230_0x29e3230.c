// Function: sub_29E3230
// Address: 0x29e3230
//
void __fastcall sub_29E3230(__int64 a1, __int64 a2, __int64 *a3, __int64 *a4, __int64 a5)
{
  int v6; // eax
  __int64 v7; // rdx
  __int64 v8; // rsi
  unsigned int v9; // ecx
  _QWORD *v10; // rax
  __int64 v11; // rdi
  unsigned __int64 v12; // r13
  __int64 v13; // rax
  int v14; // eax
  __int64 v15; // rbx
  unsigned __int64 v16; // r8
  __int64 v17; // r14
  __int64 v18; // rax
  _BYTE *v19; // rsi
  unsigned __int64 v20; // r13
  unsigned __int64 *v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r9
  __int64 v25; // r10
  unsigned __int64 *v26; // rax
  unsigned __int64 v27; // rax
  __int64 v28; // rax
  char v29; // dl
  int v30; // r9d
  unsigned __int64 v34; // [rsp+20h] [rbp-100h]
  __int64 v35; // [rsp+20h] [rbp-100h]
  unsigned __int64 v37[2]; // [rsp+30h] [rbp-F0h] BYREF
  unsigned __int64 v38; // [rsp+40h] [rbp-E0h]
  __int64 v39; // [rsp+50h] [rbp-D0h] BYREF
  unsigned __int64 *v40; // [rsp+58h] [rbp-C8h]
  __int64 v41; // [rsp+60h] [rbp-C0h]
  int v42; // [rsp+68h] [rbp-B8h]
  char v43; // [rsp+6Ch] [rbp-B4h]
  char v44; // [rsp+70h] [rbp-B0h] BYREF

  v40 = (unsigned __int64 *)&v44;
  v6 = *(_DWORD *)(a2 + 16);
  v39 = 0;
  v41 = 16;
  v42 = 0;
  v43 = 1;
  if ( v6 )
  {
    v15 = *(_QWORD *)(a2 + 8);
    v16 = (unsigned __int64)*(unsigned int *)(a2 + 24) << 6;
    v17 = v15 + v16;
    if ( v15 + v16 != v15 )
    {
      while ( 1 )
      {
        v18 = *(_QWORD *)(v15 + 24);
        if ( v18 != -8192 && v18 != -4096 )
          break;
        v15 += 64;
        if ( v17 == v15 )
          goto LABEL_2;
      }
      if ( v17 != v15 )
      {
        while ( 1 )
        {
          v19 = *(_BYTE **)(v15 + 24);
          if ( *v19 == 23 )
          {
            v20 = *(_QWORD *)(v15 + 56);
            if ( v20 )
              break;
          }
LABEL_31:
          v15 += 64;
          if ( v15 != v17 )
          {
            while ( 1 )
            {
              v28 = *(_QWORD *)(v15 + 24);
              if ( v28 != -8192 && v28 != -4096 )
                break;
              v15 += 64;
              if ( v17 == v15 )
                goto LABEL_2;
            }
            if ( v17 != v15 )
              continue;
          }
          goto LABEL_2;
        }
        v25 = sub_FDD860(a4, (__int64)v19);
        if ( v43 )
        {
          v26 = v40;
          v21 = &v40[HIDWORD(v41)];
          if ( v40 != v21 )
          {
            while ( v20 != *v26 )
            {
              if ( v21 == ++v26 )
                goto LABEL_40;
            }
            goto LABEL_28;
          }
LABEL_40:
          if ( HIDWORD(v41) < (unsigned int)v41 )
          {
            ++HIDWORD(v41);
            *v21 = v20;
            ++v39;
            goto LABEL_30;
          }
        }
        v35 = v25;
        sub_C8CC70((__int64)&v39, v20, (__int64)v21, v22, v23, v24);
        v25 = v35;
        if ( v29 )
        {
LABEL_30:
          sub_FE1040(a3, v20, v25);
          goto LABEL_31;
        }
LABEL_28:
        v34 = v25;
        v27 = sub_FDD860(a3, v20);
        v25 = v34;
        if ( v34 < v27 )
          v25 = v27;
        goto LABEL_30;
      }
    }
  }
LABEL_2:
  v7 = *(unsigned int *)(a2 + 24);
  if ( !(_DWORD)v7 )
    goto LABEL_14;
  v8 = *(_QWORD *)(a2 + 8);
  v9 = (v7 - 1) & (((unsigned int)a5 >> 9) ^ ((unsigned int)a5 >> 4));
  v10 = (_QWORD *)(v8 + ((unsigned __int64)v9 << 6));
  v11 = v10[3];
  if ( a5 != v11 )
  {
    v14 = 1;
    while ( v11 != -4096 )
    {
      v30 = v14 + 1;
      v9 = (v7 - 1) & (v14 + v9);
      v10 = (_QWORD *)(v8 + ((unsigned __int64)v9 << 6));
      v11 = v10[3];
      if ( a5 == v11 )
        goto LABEL_4;
      v14 = v30;
    }
    goto LABEL_14;
  }
LABEL_4:
  if ( v10 == (_QWORD *)(v8 + (v7 << 6)) )
  {
LABEL_14:
    v12 = 0;
    goto LABEL_10;
  }
  v37[0] = 6;
  v12 = v10[7];
  v37[1] = 0;
  v38 = v12;
  if ( v12 != -4096 && v12 != 0 && v12 != -8192 )
  {
    sub_BD6050(v37, v10[5] & 0xFFFFFFFFFFFFFFF8LL);
    v12 = v38;
    if ( v38 != 0 && v38 != -4096 && v38 != -8192 )
      sub_BD60C0(v37);
  }
LABEL_10:
  v13 = sub_FDD860(a3, a1);
  sub_FE1050(a3, v12, v13, (__int64)&v39);
  if ( !v43 )
    _libc_free((unsigned __int64)v40);
}
