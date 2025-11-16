// Function: sub_2AB5110
// Address: 0x2ab5110
//
__int64 __fastcall sub_2AB5110(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 v4; // rcx
  int v5; // eax
  int v7; // esi
  unsigned int v9; // edx
  __int64 *v10; // rax
  __int64 v11; // rdi
  unsigned int *v12; // r12
  unsigned __int8 *v13; // r15
  __int64 *v14; // rbx
  int v15; // r14d
  unsigned int v16; // esi
  __int64 v17; // r11
  int v18; // ebx
  unsigned int v19; // edi
  unsigned int v20; // r9d
  int v21; // edx
  unsigned int v22; // ecx
  unsigned int *v23; // rax
  __int64 v24; // r8
  __int64 v25; // rax
  unsigned __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // rdx
  unsigned int v29; // edi
  unsigned int v30; // esi
  int v31; // r8d
  __int64 *v32; // r14
  __int64 v33; // r14
  int v35; // eax
  int i; // eax
  __int64 v37; // rax
  __int64 v38; // rdx
  bool v39; // of
  int v40; // eax
  int v41; // r8d
  __int64 v42; // [rsp+10h] [rbp-B0h]
  int v43; // [rsp+18h] [rbp-A8h]
  __int64 v44; // [rsp+20h] [rbp-A0h]
  __int64 v47; // [rsp+48h] [rbp-78h]
  _QWORD v48[2]; // [rsp+50h] [rbp-70h] BYREF
  __int64 v49; // [rsp+60h] [rbp-60h] BYREF
  int v50; // [rsp+68h] [rbp-58h]
  _BYTE *v51; // [rsp+70h] [rbp-50h] BYREF
  __int64 v52; // [rsp+78h] [rbp-48h]
  _BYTE v53[64]; // [rsp+80h] [rbp-40h] BYREF

  v3 = *(_QWORD *)(a1 + 504);
  v4 = *(_QWORD *)(v3 + 56);
  v5 = *(_DWORD *)(v3 + 72);
  if ( !v5 )
    goto LABEL_39;
  v7 = v5 - 1;
  v9 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v10 = (__int64 *)(v4 + 16LL * v9);
  v11 = *v10;
  if ( a2 != *v10 )
  {
    v40 = 1;
    while ( v11 != -4096 )
    {
      v41 = v40 + 1;
      v9 = v7 & (v40 + v9);
      v10 = (__int64 *)(v4 + 16LL * v9);
      v11 = *v10;
      if ( a2 == *v10 )
        goto LABEL_3;
      v40 = v41;
    }
LABEL_39:
    BUG();
  }
LABEL_3:
  v12 = (unsigned int *)v10[1];
  v13 = (unsigned __int8 *)*((_QWORD *)v12 + 6);
  if ( *v13 == 61 )
    v14 = (__int64 *)*((_QWORD *)v13 + 1);
  else
    v14 = *(__int64 **)(*((_QWORD *)v13 - 8) + 8LL);
  v42 = sub_2AAEDF0((__int64)v14, a3);
  v15 = *v12;
  LODWORD(v47) = *v12 * a3;
  BYTE4(v47) = BYTE4(a3);
  v44 = sub_BCE1B0(v14, v47);
  v51 = v53;
  v52 = 0x400000000LL;
  if ( v15 )
  {
    v16 = v12[10];
    v17 = *((_QWORD *)v12 + 2);
    v18 = 0;
    v19 = v12[8];
    do
    {
      v20 = v16;
      v21 = v16 + v18;
      if ( v19 )
      {
        v22 = (v19 - 1) & (37 * v21);
        v23 = (unsigned int *)(v17 + 16LL * v22);
        v24 = *v23;
        if ( v21 == (_DWORD)v24 )
        {
LABEL_9:
          if ( *((_QWORD *)v23 + 1) )
          {
            v25 = (unsigned int)v52;
            v26 = (unsigned int)v52 + 1LL;
            if ( v26 > HIDWORD(v52) )
            {
              sub_C8D5F0((__int64)&v51, v53, v26, 4u, v24, v16);
              v25 = (unsigned int)v52;
            }
            *(_DWORD *)&v51[4 * v25] = v18;
            v16 = v12[10];
            LODWORD(v52) = v52 + 1;
            v17 = *((_QWORD *)v12 + 2);
            v19 = v12[8];
            v20 = v16;
          }
        }
        else
        {
          v35 = 1;
          while ( (_DWORD)v24 != 0x7FFFFFFF )
          {
            v22 = (v19 - 1) & (v35 + v22);
            v43 = v35 + 1;
            v23 = (unsigned int *)(v17 + 16LL * v22);
            v24 = *v23;
            if ( v21 == (_DWORD)v24 )
              goto LABEL_9;
            v35 = v43;
          }
        }
      }
      ++v18;
    }
    while ( v18 != v15 );
  }
  else
  {
    v20 = v12[10];
    v17 = *((_QWORD *)v12 + 2);
    v19 = v12[8];
  }
  v27 = *v12;
  v28 = v20 + (unsigned int)v27 - 1;
  if ( v19 )
  {
    v29 = v19 - 1;
    v30 = v29 & (37 * v28);
    v31 = *(_DWORD *)(v17 + 16LL * v30);
    if ( v31 != (_DWORD)v28 )
    {
      for ( i = 1; v31 != 0x7FFFFFFF; ++i )
      {
        v30 = v29 & (i + v30);
        v31 = *(_DWORD *)(v17 + 16LL * v30);
        if ( (_DWORD)v28 == v31 )
          break;
      }
    }
  }
  v32 = *(__int64 **)(a1 + 448);
  sub_B19060(*(_QWORD *)(a1 + 440) + 440LL, a2, v28, v27);
  v33 = sub_DFD610(v32, (unsigned int)*v13 - 29, v44, *v12, (__int64)v51);
  if ( *((_BYTE *)v12 + 4) )
  {
    v48[0] = sub_DFBC30(*(__int64 **)(a1 + 448), 1, v42, 0, 0, *(unsigned int *)(a1 + 992), 0, 0, 0, 0, 0);
    v37 = v12[6];
    v48[1] = v38;
    v49 = v37;
    v50 = 0;
    sub_2AA9150((__int64)&v49, (__int64)v48);
    v39 = __OFADD__(v49, v33);
    v33 += v49;
    if ( v39 )
    {
      v33 = 0x7FFFFFFFFFFFFFFFLL;
      if ( v49 <= 0 )
        v33 = 0x8000000000000000LL;
    }
  }
  if ( v51 != v53 )
    _libc_free((unsigned __int64)v51);
  return v33;
}
