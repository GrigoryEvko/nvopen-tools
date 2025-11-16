// Function: sub_3393C00
// Address: 0x3393c00
//
void __fastcall sub_3393C00(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  __int64 v7; // r14
  int v8; // eax
  __int64 v9; // r15
  __int64 v10; // rbx
  __int64 *v11; // rax
  __int64 v12; // rcx
  __int64 *v13; // rdx
  char v14; // dl
  __int64 v15; // r14
  __int64 v16; // rsi
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  int v21; // r9d
  __int64 v22; // rdx
  __int128 v23; // rcx
  __int64 v24; // rax
  __int64 v25; // rsi
  __int64 v26; // rax
  int v27; // edx
  __int64 v28; // rbx
  int v29; // r12d
  __int64 v30; // [rsp+0h] [rbp-1A0h]
  __int64 v31; // [rsp+10h] [rbp-190h]
  __int128 v32; // [rsp+10h] [rbp-190h]
  __int64 v33; // [rsp+40h] [rbp-160h] BYREF
  int v34; // [rsp+48h] [rbp-158h]
  __int64 v35; // [rsp+50h] [rbp-150h] BYREF
  __int64 *v36; // [rsp+58h] [rbp-148h]
  __int64 v37; // [rsp+60h] [rbp-140h]
  int v38; // [rsp+68h] [rbp-138h]
  char v39; // [rsp+6Ch] [rbp-134h]
  char v40; // [rsp+70h] [rbp-130h] BYREF

  v6 = *(_QWORD *)(a1 + 960);
  v35 = 0;
  v37 = 32;
  v7 = *(_QWORD *)(v6 + 744);
  v38 = 0;
  v36 = (__int64 *)&v40;
  LODWORD(v6) = *(_DWORD *)(a2 + 4);
  v39 = 1;
  v8 = v6 & 0x7FFFFFF;
  if ( v8 != 1 )
  {
    v9 = 32;
    v10 = *(_QWORD *)(*(_QWORD *)(a2 - 8) + 32LL);
    v31 = 32 * ((unsigned int)(v8 - 2) + 2LL);
LABEL_3:
    v11 = v36;
    v12 = HIDWORD(v37);
    v13 = &v36[HIDWORD(v37)];
    if ( v36 == v13 )
    {
LABEL_21:
      if ( HIDWORD(v37) >= (unsigned int)v37 )
        goto LABEL_9;
      ++HIDWORD(v37);
      *v13 = v10;
      ++v35;
LABEL_10:
      v9 += 32;
      sub_3373E10(
        a1,
        v7,
        *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 960) + 56LL) + 8LL * *(unsigned int *)(v10 + 44)),
        0xFFFFFFFFLL,
        a5,
        a6);
      if ( v9 != v31 )
        goto LABEL_8;
    }
    else
    {
      while ( v10 != *v11 )
      {
        if ( v13 == ++v11 )
          goto LABEL_21;
      }
      while ( 1 )
      {
        v9 += 32;
        if ( v9 == v31 )
          break;
LABEL_8:
        v13 = *(__int64 **)(a2 - 8);
        v10 = v13[(unsigned __int64)v9 / 8];
        if ( v39 )
          goto LABEL_3;
LABEL_9:
        sub_C8CC70((__int64)&v35, v10, (__int64)v13, v12, a5, a6);
        if ( v14 )
          goto LABEL_10;
      }
    }
  }
  sub_2E33470(*(unsigned int **)(v7 + 144), *(unsigned int **)(v7 + 152));
  v15 = *(_QWORD *)(a1 + 864);
  v16 = **(_QWORD **)(a2 - 8);
  *(_QWORD *)&v32 = sub_338B750(a1, v16);
  *((_QWORD *)&v32 + 1) = v17;
  v33 = 0;
  *(_QWORD *)&v23 = sub_3373A60(a1, v16, v17, v18, v19, v20);
  *((_QWORD *)&v23 + 1) = v22;
  v24 = *(_QWORD *)a1;
  v34 = *(_DWORD *)(a1 + 848);
  if ( v24 )
  {
    if ( &v33 != (__int64 *)(v24 + 48) )
    {
      v25 = *(_QWORD *)(v24 + 48);
      v33 = v25;
      if ( v25 )
      {
        v30 = v23;
        sub_B96E90((__int64)&v33, v25, 1);
        *(_QWORD *)&v23 = v30;
      }
    }
  }
  v26 = sub_3406EB0(v15, 302, (unsigned int)&v33, 1, 0, v21, v23, v32);
  v28 = v26;
  v29 = v27;
  if ( v26 )
  {
    nullsub_1875(v26, v15, 0);
    *(_QWORD *)(v15 + 384) = v28;
    *(_DWORD *)(v15 + 392) = v29;
    sub_33E2B60(v15, 0);
  }
  else
  {
    *(_QWORD *)(v15 + 384) = 0;
    *(_DWORD *)(v15 + 392) = v27;
  }
  if ( v33 )
    sub_B91220((__int64)&v33, v33);
  if ( !v39 )
    _libc_free((unsigned __int64)v36);
}
