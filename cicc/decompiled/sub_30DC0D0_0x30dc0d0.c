// Function: sub_30DC0D0
// Address: 0x30dc0d0
//
__int64 __fastcall sub_30DC0D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v8; // r13d
  __int64 v9; // r15
  unsigned int v10; // r14d
  __int64 v11; // r14
  __int64 **v12; // r10
  __int64 v13; // r14
  __int64 v14; // r15
  __int64 v15; // r14
  __int64 v16; // r14
  __int64 v17; // r11
  __int64 v18; // rbx
  unsigned __int8 **v19; // rcx
  int v20; // eax
  unsigned __int8 **v21; // rdx
  unsigned __int64 v22; // rax
  __int64 v23; // r12
  int v24; // edx
  int v25; // ebx
  int v27; // eax
  __int64 v28; // rcx
  int v29; // esi
  unsigned int v30; // edx
  __int64 *v31; // rax
  __int64 v32; // rdi
  __int64 v33; // rcx
  __int64 *v34; // rax
  bool v35; // cc
  int v36; // eax
  int v37; // r8d
  __int64 **v38; // [rsp+8h] [rbp-88h]
  __int64 v39; // [rsp+28h] [rbp-68h] BYREF
  unsigned __int8 **v40; // [rsp+30h] [rbp-60h] BYREF
  unsigned __int64 v41; // [rsp+38h] [rbp-58h] BYREF
  _DWORD v42[20]; // [rsp+40h] [rbp-50h] BYREF

  v8 = sub_30D92D0(a1, a2, a3, a4, a5, a6);
  if ( (_BYTE)v8 )
    return v8;
  v9 = *(_QWORD *)(a2 - 32);
  v10 = sub_BCB060(*(_QWORD *)(v9 + 8));
  if ( (unsigned int)sub_AE43A0(*(_QWORD *)(a1 + 80), *(_QWORD *)(a2 + 8)) >= v10 )
  {
    v27 = *(_DWORD *)(a1 + 256);
    v28 = *(_QWORD *)(a1 + 240);
    if ( v27 )
    {
      v29 = v27 - 1;
      v30 = (v27 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
      v31 = (__int64 *)(v28 + 32LL * v30);
      v32 = *v31;
      if ( v9 != *v31 )
      {
        v36 = 1;
        while ( v32 != -4096 )
        {
          v37 = v36 + 1;
          v30 = v29 & (v36 + v30);
          v31 = (__int64 *)(v28 + 32LL * v30);
          v32 = *v31;
          if ( v9 == *v31 )
            goto LABEL_21;
          v36 = v37;
        }
        goto LABEL_3;
      }
LABEL_21:
      v33 = v31[1];
      v40 = (unsigned __int8 **)v33;
      v42[0] = *((_DWORD *)v31 + 6);
      if ( v42[0] > 0x40u )
      {
        sub_C43780((__int64)&v41, (const void **)v31 + 2);
        if ( !v40 )
          goto LABEL_26;
        goto LABEL_23;
      }
      v41 = v31[2];
      if ( v33 )
      {
LABEL_23:
        v39 = a2;
        v34 = sub_30DA4E0(a1 + 232, &v39);
        v35 = *((_DWORD *)v34 + 4) <= 0x40u;
        *v34 = (__int64)v40;
        if ( v35 && v42[0] <= 0x40u )
        {
          v34[1] = v41;
          *((_DWORD *)v34 + 4) = v42[0];
        }
        else
        {
          sub_C43990((__int64)(v34 + 1), (__int64)&v41);
        }
LABEL_26:
        if ( v42[0] > 0x40u && v41 )
          j_j___libc_free_0_0(v41);
      }
    }
  }
LABEL_3:
  v11 = sub_30D1740(a1, v9);
  if ( v11 )
  {
    v40 = (unsigned __int8 **)a2;
    *sub_30DA630(a1 + 168, (__int64 *)&v40) = v11;
  }
  v12 = *(__int64 ***)(a1 + 8);
  v13 = 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
  {
    v14 = *(_QWORD *)(a2 - 8);
    v15 = v14 + v13;
  }
  else
  {
    v14 = a2 - v13;
    v15 = a2;
  }
  v16 = v15 - v14;
  v40 = (unsigned __int8 **)v42;
  v17 = v16 >> 5;
  v41 = 0x400000000LL;
  v18 = v16 >> 5;
  if ( (unsigned __int64)v16 > 0x80 )
  {
    v38 = v12;
    sub_C8D5F0((__int64)&v40, v42, v16 >> 5, 8u, (__int64)&v40, (__int64)v42);
    v21 = v40;
    v20 = v41;
    v17 = v16 >> 5;
    v12 = v38;
    v19 = &v40[(unsigned int)v41];
  }
  else
  {
    v19 = (unsigned __int8 **)v42;
    v20 = 0;
    v21 = (unsigned __int8 **)v42;
  }
  if ( v16 > 0 )
  {
    v22 = 0;
    do
    {
      v19[v22 / 8] = *(unsigned __int8 **)(v14 + 4 * v22);
      v22 += 8LL;
      --v18;
    }
    while ( v18 );
    v21 = v40;
    v20 = v41;
  }
  LODWORD(v41) = v17 + v20;
  v23 = sub_DFCEF0(v12, (unsigned __int8 *)a2, v21, (unsigned int)(v17 + v20), 3);
  v25 = v24;
  if ( v40 != (unsigned __int8 **)v42 )
    _libc_free((unsigned __int64)v40);
  if ( !v25 )
    LOBYTE(v8) = v23 == 0;
  return v8;
}
