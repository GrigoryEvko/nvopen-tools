// Function: sub_2A31020
// Address: 0x2a31020
//
_BYTE **__fastcall sub_2A31020(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // r8
  __int64 v7; // r15
  __int64 v8; // rdx
  _BYTE **result; // rax
  int v10; // r14d
  unsigned int v11; // r13d
  __int64 v12; // rax
  unsigned int v13; // esi
  _QWORD *v14; // rax
  __int64 v15; // r9
  __int64 v16; // r14
  __int64 v17; // rbx
  unsigned int v18; // r12d
  _QWORD *v19; // rax
  _BYTE *v20; // r13
  _BYTE *v21; // r12
  unsigned int v22; // esi
  unsigned int v23; // eax
  __int64 v24; // rax
  unsigned __int64 v25; // rdx
  bool v27; // [rsp+17h] [rbp-99h]
  __int64 v28; // [rsp+18h] [rbp-98h]
  unsigned int v30; // [rsp+38h] [rbp-78h]
  unsigned int v31; // [rsp+3Ch] [rbp-74h]
  _QWORD *v32; // [rsp+40h] [rbp-70h] BYREF
  unsigned int v33; // [rsp+48h] [rbp-68h]
  _BYTE *v34; // [rsp+50h] [rbp-60h] BYREF
  __int64 v35; // [rsp+58h] [rbp-58h]
  _BYTE v36[80]; // [rsp+60h] [rbp-50h] BYREF

  v7 = sub_AA5930(a1);
  v28 = v8;
  result = &v34;
  v27 = a3 == 0;
  while ( v28 != v7 )
  {
    v10 = *(_DWORD *)(v7 + 4) & 0x7FFFFFF;
    v33 = *(_DWORD *)(a4 + 8);
    if ( v33 > 0x40 )
    {
      sub_C43780((__int64)&v32, (const void **)a4);
      if ( !v10 )
        goto LABEL_35;
    }
    else
    {
      v32 = *(_QWORD **)a4;
      if ( !v10 )
        goto LABEL_35;
    }
    if ( !v27 )
    {
      v11 = 0;
      while ( 1 )
      {
        v12 = v11;
        v13 = v11++;
        v14 = (_QWORD *)(*(_QWORD *)(v7 - 8) + 32LL * *(unsigned int *)(v7 + 72) + 8 * v12);
        if ( a2 == *v14 )
          break;
        if ( v10 == v11 )
          goto LABEL_9;
      }
      v11 = v13;
      *v14 = a3;
LABEL_9:
      ++v11;
      goto LABEL_10;
    }
LABEL_35:
    v11 = 0;
    if ( a3 )
      goto LABEL_9;
LABEL_10:
    v15 = v11;
    v31 = v10;
    v16 = a2;
    v30 = 0;
    v34 = v36;
    v17 = 8LL * v11;
    v18 = v33;
    v35 = 0x800000000LL;
    if ( v33 <= 0x40 )
    {
LABEL_11:
      v19 = v32;
      goto LABEL_12;
    }
    while ( 1 )
    {
      v23 = sub_C444A0((__int64)&v32);
      v15 = v23;
      if ( v18 - v23 <= 0x40 )
      {
        v19 = (_QWORD *)*v32;
LABEL_12:
        if ( !v19 )
          break;
      }
      if ( v11 >= v31 )
        break;
      if ( v16 == *(_QWORD *)(*(_QWORD *)(v7 - 8) + 32LL * *(unsigned int *)(v7 + 72) + v17) )
      {
        v24 = v30;
        v25 = v30 + 1LL;
        if ( v25 > HIDWORD(v35) )
        {
          sub_C8D5F0((__int64)&v34, v36, v25, 4u, v6, v15);
          v24 = (unsigned int)v35;
        }
        *(_DWORD *)&v34[4 * v24] = v11;
        LODWORD(v35) = v35 + 1;
        sub_C46F20((__int64)&v32, 1u);
        v18 = v33;
        v30 = v35;
      }
      ++v11;
      v17 += 8;
      if ( v18 <= 0x40 )
        goto LABEL_11;
    }
    v20 = v34;
    a2 = v16;
    v21 = &v34[4 * v30];
    if ( v34 != v21 )
    {
      do
      {
        v22 = *((_DWORD *)v21 - 1);
        v21 -= 4;
        sub_B48BF0(v7, v22, 1);
      }
      while ( v20 != v21 );
      v21 = v34;
    }
    if ( v21 != v36 )
      _libc_free((unsigned __int64)v21);
    if ( v33 > 0x40 && v32 )
      j_j___libc_free_0_0((unsigned __int64)v32);
    result = *(_BYTE ***)(v7 + 32);
    if ( !result )
      BUG();
    v7 = 0;
    if ( *((_BYTE *)result - 24) == 84 )
      v7 = (__int64)(result - 3);
  }
  return result;
}
