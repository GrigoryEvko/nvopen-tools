// Function: sub_B9E560
// Address: 0xb9e560
//
__int64 __fastcall sub_B9E560(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v6; // rsi
  __int64 v7; // r12
  unsigned int *v8; // r14
  unsigned int v9; // r12d
  _BYTE *v10; // rbx
  __int64 v11; // rax
  __int64 v12; // rcx
  void *v13; // r8
  void *v14; // r11
  _QWORD *v15; // r15
  void *v16; // r11
  __int64 *v17; // rax
  __int64 *v18; // rax
  __int64 result; // rax
  _BYTE *v20; // rdi
  __int64 v21; // rax
  __int64 v22; // r8
  __int64 v23; // r15
  _BYTE *v24; // rdi
  __int64 v25; // rbx
  _BYTE *v26; // rax
  __int64 v27; // r8
  unsigned int v28; // eax
  __int64 v29; // rax
  _QWORD *v30; // rax
  __int64 *v31; // rax
  __int64 v32; // rax
  __int64 v33; // rax
  void *dest; // [rsp+10h] [rbp-110h]
  __int64 v35; // [rsp+18h] [rbp-108h]
  __int64 v36; // [rsp+18h] [rbp-108h]
  void *src; // [rsp+20h] [rbp-100h]
  void *srca; // [rsp+20h] [rbp-100h]
  __int64 v39; // [rsp+28h] [rbp-F8h]
  __int64 v40; // [rsp+28h] [rbp-F8h]
  _QWORD *v41; // [rsp+28h] [rbp-F8h]
  _QWORD *v42; // [rsp+28h] [rbp-F8h]
  size_t n; // [rsp+30h] [rbp-F0h]
  size_t na; // [rsp+30h] [rbp-F0h]
  unsigned int *v45; // [rsp+38h] [rbp-E8h]
  __int64 v46; // [rsp+40h] [rbp-E0h] BYREF
  unsigned int v47; // [rsp+48h] [rbp-D8h]
  _QWORD *v48; // [rsp+50h] [rbp-D0h] BYREF
  size_t v49; // [rsp+58h] [rbp-C8h]
  unsigned int *v50; // [rsp+60h] [rbp-C0h] BYREF
  __int64 v51; // [rsp+68h] [rbp-B8h]
  _BYTE v52[176]; // [rsp+70h] [rbp-B0h] BYREF

  v6 = (__int64)&v50;
  v50 = (unsigned int *)v52;
  v51 = 0x800000000LL;
  sub_B9A9D0(a2, (__int64)&v50);
  v7 = 4LL * (unsigned int)v51;
  v45 = &v50[v7];
  if ( v50 == &v50[v7] )
    goto LABEL_15;
  v8 = v50;
  v9 = a3;
  do
  {
    v6 = *v8;
    v10 = (_BYTE *)*((_QWORD *)v8 + 1);
    if ( !v9 )
      goto LABEL_12;
    if ( (_DWORD)v6 != 19 )
    {
      if ( (_DWORD)v6 )
      {
LABEL_12:
        sub_B994D0(a1, v6, (__int64)v10);
        goto LABEL_13;
      }
      if ( *v10 == 25
        || (v20 = v10 - 16, v10 = *(_BYTE **)sub_A17150(v10 - 16), (v21 = *((_QWORD *)sub_A17150(v20) + 1)) == 0) )
      {
        v11 = sub_22077B0(16);
        v12 = 0;
        v13 = 0;
        n = 16;
        v14 = (void *)(v11 + 16);
        v15 = (_QWORD *)v11;
        v39 = v11 + 16;
      }
      else
      {
        v22 = *(_QWORD *)(v21 + 16);
        v23 = *(_QWORD *)(v21 + 24) - v22;
        if ( (unsigned __int64)((v23 >> 3) + 2) > 0xFFFFFFFFFFFFFFFLL )
          sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
        if ( v23 >> 3 == -2 )
        {
          MEMORY[0] = 0;
          BUG();
        }
        v36 = (*(_QWORD *)(v21 + 24) - v22) >> 3;
        srca = *(void **)(v21 + 16);
        n = v23 + 16;
        v33 = sub_22077B0(v23 + 16);
        v13 = srca;
        v12 = v36;
        v15 = (_QWORD *)v33;
        v39 = v33 + n;
        if ( v33 + n == v33 )
        {
          v16 = (void *)(v39 + 16);
          goto LABEL_9;
        }
        v14 = (void *)(v33 + 16);
      }
      v6 = 0;
      dest = v14;
      v35 = v12;
      src = v13;
      memset(v15, 0, n);
      v16 = dest;
      v12 = v35;
      v13 = src;
LABEL_9:
      *v15 = 35;
      v15[1] = v9;
      if ( 8 * v12 )
      {
        v6 = (__int64)v13;
        memmove(v16, v13, 8 * v12);
      }
      v17 = (__int64 *)sub_BD5C60(a1, v6);
      v40 = sub_B0D000(v17, v15, (v39 - (__int64)v15) >> 3, 0, 1);
      v18 = (__int64 *)sub_BD5C60(a1, v15);
      v10 = (_BYTE *)sub_B0EF30(v18, (__int64)v10, v40, 0, 1);
      j_j___libc_free_0(v15, n);
      v6 = *v8;
      goto LABEL_12;
    }
    v24 = v10 - 16;
    v25 = *(_QWORD *)(*(_QWORD *)sub_A17150(v10 - 16) + 136LL);
    v26 = sub_A17150(v24);
    v27 = v9;
    na = *((_QWORD *)v26 + 1);
    v47 = *(_DWORD *)(v25 + 32);
    if ( v47 > 0x40 )
    {
      sub_C43780(&v46, v25 + 24);
      v27 = v9;
    }
    else
    {
      v46 = *(_QWORD *)(v25 + 24);
    }
    sub_C46A40(&v46, v27);
    v28 = v47;
    v47 = 0;
    LODWORD(v49) = v28;
    v48 = (_QWORD *)v46;
    v29 = sub_AD8D80(*(_QWORD *)(v25 + 8), (__int64)&v48);
    v30 = sub_B98A20(v29, (__int64)&v48);
    if ( (unsigned int)v49 > 0x40 && v48 )
    {
      v41 = v30;
      j_j___libc_free_0_0(v48);
      v30 = v41;
    }
    if ( v47 > 0x40 && v46 )
    {
      v42 = v30;
      j_j___libc_free_0_0(v46);
      v30 = v42;
    }
    v48 = v30;
    v49 = na;
    v31 = (__int64 *)sub_BD5C60(a1, &v48);
    v32 = sub_B9C770(v31, (__int64 *)&v48, (__int64 *)2, 0, 1);
    v6 = 19;
    sub_B994D0(a1, 19, v32);
LABEL_13:
    v8 += 4;
  }
  while ( v45 != v8 );
  v45 = v50;
LABEL_15:
  result = (__int64)v45;
  if ( v45 != (unsigned int *)v52 )
    return _libc_free(v45, v6);
  return result;
}
