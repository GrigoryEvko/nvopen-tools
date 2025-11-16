// Function: sub_37F1600
// Address: 0x37f1600
//
_QWORD *__fastcall sub_37F1600(_QWORD *a1, _QWORD *a2, __int64 a3)
{
  unsigned __int64 *v4; // rdi
  unsigned __int64 v5; // r10
  __int64 *v6; // r8
  __int64 v7; // r11
  __int64 v8; // rax
  __int64 v9; // r12
  _QWORD *v10; // r14
  __int64 v11; // r9
  __int64 v12; // rax
  unsigned __int64 **v13; // rax
  unsigned __int64 *v14; // r13
  _QWORD *v15; // rbx
  __int64 v16; // rax
  unsigned __int64 v17; // rcx
  __int64 v18; // rsi
  unsigned __int64 *v19; // rdx
  _QWORD *v20; // r15
  __int64 v21; // rdx
  __int64 v22; // rsi
  unsigned __int64 *v23; // rax
  _QWORD *v24; // rax
  unsigned __int64 *v25; // r14
  _QWORD *v26; // r13
  _QWORD *v27; // r15
  __int64 v28; // rdx
  __int64 v29; // r10
  _QWORD *v30; // rax
  __int64 v31; // rdi
  __int64 v32; // rax
  __int64 v33; // rdx
  __int64 *v34; // rcx
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // rax
  __int64 v39; // rdx
  __int64 v40; // [rsp+0h] [rbp-D0h]
  __int64 v41; // [rsp+8h] [rbp-C8h]
  __int64 v42; // [rsp+10h] [rbp-C0h]
  __int64 v43; // [rsp+18h] [rbp-B8h]
  unsigned __int64 *v44; // [rsp+28h] [rbp-A8h]
  __int64 v45; // [rsp+30h] [rbp-A0h]
  __int64 v47[4]; // [rsp+40h] [rbp-90h] BYREF
  _QWORD v48[4]; // [rsp+60h] [rbp-70h] BYREF
  _QWORD v49[10]; // [rsp+80h] [rbp-50h] BYREF

  v4 = a1 + 14;
  v5 = v4[2];
  v6 = (__int64 *)v4[5];
  v7 = a1[17];
  v44 = v4;
  v8 = ((((__int64)(v4[9] - (_QWORD)v6) >> 3) - 1) << 6)
     + ((__int64)(v4[6] - v4[7]) >> 3)
     + ((__int64)(v4[4] - v5) >> 3);
  if ( (_DWORD)v8 )
  {
    v9 = 0;
    v10 = a1;
    v43 = (unsigned int)v8;
    v42 = 16 * a3;
    v45 = (16 * a3) >> 4;
    v40 = (__int64)(a1 + 2);
    while ( 1 )
    {
      v11 = (__int64)(v5 - v7) >> 3;
      v12 = v11 + v9;
      if ( v11 + v9 < 0 )
      {
        v39 = ~((unsigned __int64)~v12 >> 6);
      }
      else
      {
        if ( v12 <= 63 )
        {
          v13 = (unsigned __int64 **)(v5 + 8 * v9);
          goto LABEL_6;
        }
        v39 = v12 >> 6;
      }
      v13 = (unsigned __int64 **)(v6[v39] + 8 * (v12 - (v39 << 6)));
LABEL_6:
      v14 = *v13;
      v15 = a2;
      v16 = v45;
      v17 = *v14;
      while ( v16 > 0 )
      {
        while ( 1 )
        {
          v18 = v16 >> 1;
          v19 = &v15[2 * (v16 >> 1)];
          if ( v17 > *v19 )
            break;
          v16 >>= 1;
          if ( v18 <= 0 )
            goto LABEL_10;
        }
        v15 = v19 + 2;
        v16 = v16 - v18 - 1;
      }
LABEL_10:
      v20 = a2;
      if ( v42 > 0 )
      {
        v21 = v45;
        do
        {
          while ( 1 )
          {
            v22 = v21 >> 1;
            v23 = &v20[2 * (v21 >> 1)];
            if ( v17 < *v23 || v17 == *v23 && (v23[1] & 0x8000000000000000LL) != 0LL )
              break;
            v20 = v23 + 2;
            v21 = v21 - v22 - 1;
            if ( v21 <= 0 )
              goto LABEL_17;
          }
          v21 >>= 1;
        }
        while ( v22 > 0 );
      }
LABEL_17:
      if ( v20 == v15 )
        goto LABEL_27;
      v24 = v10;
      v25 = v14;
      v26 = v20;
      v27 = v24;
      while ( *v15 != v17 )
      {
        v15 += 2;
        if ( v15 == v26 )
          goto LABEL_26;
LABEL_20:
        v17 = *v25;
      }
      v28 = v27[2];
      v29 = v15[1];
      v27[12] += 16LL;
      v30 = (_QWORD *)((v28 + 7) & 0xFFFFFFFFFFFFFFF8LL);
      if ( v27[3] >= (unsigned __int64)(v30 + 2) && v28 )
      {
        v27[2] = v30 + 2;
      }
      else
      {
        v41 = v29;
        v30 = (_QWORD *)sub_9D1E70(v40, 16, 16, 3);
        v29 = v41;
      }
      *v30 = v29;
      v15 += 2;
      v30[1] = v25;
      v49[0] = v30;
      sub_37F09A0(v4, v49);
      if ( v15 != v26 )
        goto LABEL_20;
LABEL_26:
      v5 = v27[16];
      v7 = v27[17];
      v10 = v27;
      v6 = (__int64 *)v27[19];
      v11 = (__int64)(v5 - v7) >> 3;
LABEL_27:
      if ( v43 == ++v9 )
      {
        v31 = v11;
        goto LABEL_29;
      }
    }
  }
  v43 = 0;
  v31 = (__int64)(v5 - v7) >> 3;
LABEL_29:
  v32 = v43 + v31;
  if ( v43 + v31 < 0 )
  {
    v33 = ~((unsigned __int64)~v32 >> 6);
  }
  else
  {
    if ( v32 <= 63 )
    {
      v35 = v5 + 8 * v43;
      v34 = v6;
      goto LABEL_33;
    }
    v33 = v32 >> 6;
  }
  v34 = &v6[v33];
  v35 = *v34 + 8 * (v32 - (v33 << 6));
LABEL_33:
  v48[0] = v35;
  v36 = *v34;
  v48[3] = v34;
  v48[1] = v36;
  v48[2] = v36 + 512;
  v49[0] = v5;
  v37 = *v6;
  v49[3] = v6;
  v49[1] = v37;
  v49[2] = v37 + 512;
  return sub_37F0F40(v47, v44, (__int64)v49, (__int64)v48);
}
