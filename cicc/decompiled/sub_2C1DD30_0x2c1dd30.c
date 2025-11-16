// Function: sub_2C1DD30
// Address: 0x2c1dd30
//
void __fastcall sub_2C1DD30(__int64 a1, __int64 a2)
{
  __int64 *v4; // r15
  __int64 v5; // r13
  __int64 *v6; // r14
  __int64 v7; // rax
  __int64 v8; // r13
  __int64 *v9; // r13
  __int64 v10; // r13
  __int64 v11; // r13
  __int64 v12; // r15
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // r8
  __int64 v16; // rdx
  unsigned __int64 v17; // r9
  __int64 v18; // rax
  __int64 v19; // r8
  _BYTE *v20; // rax
  unsigned int v21; // edx
  _BYTE *v22; // r15
  __int64 v23; // rax
  unsigned int **v24; // rdi
  __int64 v25; // rsi
  __int64 v26; // rax
  _BYTE *v27; // r13
  __int64 v28; // rax
  __int64 v29; // r13
  __int64 v30; // rax
  __int64 v31; // r9
  __int64 v32; // rdx
  unsigned __int64 v33; // r8
  __int64 v34; // r15
  __int64 *v35; // rax
  _BYTE *v36; // rcx
  __int64 v37; // r8
  unsigned int v38; // eax
  __int64 v39; // [rsp+0h] [rbp-D0h]
  __int64 v40; // [rsp+10h] [rbp-C0h]
  __int64 v41; // [rsp+10h] [rbp-C0h]
  __int64 v42; // [rsp+18h] [rbp-B8h]
  __int64 v43; // [rsp+28h] [rbp-A8h]
  __int64 v44[4]; // [rsp+30h] [rbp-A0h] BYREF
  __int16 v45; // [rsp+50h] [rbp-80h]
  _BYTE *v46; // [rsp+60h] [rbp-70h] BYREF
  __int64 v47; // [rsp+68h] [rbp-68h]
  _BYTE v48[96]; // [rsp+70h] [rbp-60h] BYREF

  v4 = *(__int64 **)(a1 + 48);
  v5 = 8LL * *(unsigned int *)(a1 + 56);
  v42 = *(_QWORD *)(a1 + 136);
  v6 = &v4[(unsigned __int64)v5 / 8];
  v7 = v5 >> 3;
  v8 = v5 >> 5;
  if ( v8 )
  {
    v9 = &v4[4 * v8];
    while ( 1 )
    {
      if ( !sub_2BFB0D0(*v4) )
        goto LABEL_8;
      if ( !sub_2BFB0D0(v4[1]) )
      {
        ++v4;
        goto LABEL_8;
      }
      if ( !sub_2BFB0D0(v4[2]) )
      {
        v4 += 2;
        goto LABEL_8;
      }
      if ( !sub_2BFB0D0(v4[3]) )
        break;
      v4 += 4;
      if ( v9 == v4 )
      {
        v7 = v6 - v4;
        goto LABEL_29;
      }
    }
    v4 += 3;
    goto LABEL_8;
  }
LABEL_29:
  if ( v7 == 2 )
    goto LABEL_37;
  if ( v7 == 3 )
  {
    if ( !sub_2BFB0D0(*v4) )
      goto LABEL_8;
    ++v4;
LABEL_37:
    if ( !sub_2BFB0D0(*v4) )
      goto LABEL_8;
    ++v4;
    goto LABEL_32;
  }
  if ( v7 != 1 )
    goto LABEL_9;
LABEL_32:
  if ( sub_2BFB0D0(*v4) )
    goto LABEL_9;
LABEL_8:
  if ( v6 == v4 )
  {
LABEL_9:
    v10 = *(unsigned int *)(a1 + 56);
    v46 = v48;
    v47 = 0x600000000LL;
    if ( (_DWORD)v10 )
    {
      v11 = 8 * v10;
      v12 = 0;
      do
      {
        v13 = *(_QWORD *)(a1 + 48);
        BYTE4(v44[0]) = 0;
        LODWORD(v44[0]) = 0;
        v14 = sub_2BFB120(a2, *(_QWORD *)(v13 + v12), (unsigned int *)v44);
        v16 = (unsigned int)v47;
        v17 = (unsigned int)v47 + 1LL;
        if ( v17 > HIDWORD(v47) )
        {
          v40 = v14;
          sub_C8D5F0((__int64)&v46, v48, (unsigned int)v47 + 1LL, 8u, v15, v17);
          v16 = (unsigned int)v47;
          v14 = v40;
        }
        v12 += 8;
        *(_QWORD *)&v46[8 * v16] = v14;
        v18 = (unsigned int)(v47 + 1);
        LODWORD(v47) = v47 + 1;
      }
      while ( v11 != v12 );
      v19 = v18 - 1;
      v20 = v46;
    }
    else
    {
      v20 = v48;
      v19 = -1;
    }
    v21 = *(_DWORD *)(a1 + 156);
    v22 = (_BYTE *)v42;
    v45 = 257;
    v23 = sub_921130(
            *(unsigned int ***)(a2 + 904),
            *(_QWORD *)(v42 + 72),
            *(_QWORD *)v20,
            (_BYTE **)v20 + 1,
            v19,
            (__int64)v44,
            v21);
    v24 = *(unsigned int ***)(a2 + 904);
    v25 = *(_QWORD *)(a2 + 8);
    v45 = 257;
    v26 = sub_B37620(v24, v25, v23, v44);
    goto LABEL_16;
  }
  if ( sub_2BFB0D0(**(_QWORD **)(a1 + 48)) )
  {
    v35 = *(__int64 **)(a1 + 48);
    BYTE4(v46) = 0;
    LODWORD(v46) = 0;
    v41 = sub_2BFB120(a2, *v35, (unsigned int *)&v46);
  }
  else
  {
    v41 = sub_2BFB640(a2, **(_QWORD **)(a1 + 48), 0);
  }
  v46 = v48;
  v47 = 0x400000000LL;
  v28 = *(unsigned int *)(a1 + 56);
  if ( (unsigned int)v28 > 1 )
  {
    v29 = 8;
    v43 = 8 * v28;
    while ( 1 )
    {
      v34 = *(_QWORD *)(*(_QWORD *)(a1 + 48) + v29);
      if ( sub_2BFB0D0(v34) )
      {
        BYTE4(v44[0]) = 0;
        LODWORD(v44[0]) = 0;
        v30 = sub_2BFB120(a2, v34, (unsigned int *)v44);
        v32 = (unsigned int)v47;
        v33 = (unsigned int)v47 + 1LL;
        if ( v33 > HIDWORD(v47) )
          goto LABEL_27;
      }
      else
      {
        v30 = sub_2BFB640(a2, v34, 0);
        v32 = (unsigned int)v47;
        v33 = (unsigned int)v47 + 1LL;
        if ( v33 > HIDWORD(v47) )
        {
LABEL_27:
          v39 = v30;
          sub_C8D5F0((__int64)&v46, v48, v33, 8u, v33, v31);
          v32 = (unsigned int)v47;
          v30 = v39;
        }
      }
      v29 += 8;
      *(_QWORD *)&v46[8 * v32] = v30;
      LODWORD(v47) = v47 + 1;
      if ( v43 == v29 )
      {
        v36 = v46;
        v37 = (unsigned int)v47;
        goto LABEL_40;
      }
    }
  }
  v36 = v48;
  v37 = 0;
LABEL_40:
  v38 = *(_DWORD *)(a1 + 156);
  v22 = (_BYTE *)v42;
  v45 = 257;
  v26 = sub_921130(*(unsigned int ***)(a2 + 904), *(_QWORD *)(v42 + 72), v41, (_BYTE **)v36, v37, (__int64)v44, v38);
LABEL_16:
  v27 = (_BYTE *)v26;
  sub_2BF26E0(a2, a1 + 96, v26, 0);
  sub_2BF08A0(a2, v27, v22);
  if ( v46 != v48 )
    _libc_free((unsigned __int64)v46);
}
