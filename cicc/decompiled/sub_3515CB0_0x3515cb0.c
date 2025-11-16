// Function: sub_3515CB0
// Address: 0x3515cb0
//
__int64 __fastcall sub_3515CB0(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r14
  __int64 v7; // rbx
  __int64 *v8; // r15
  __int64 *v9; // r12
  __int64 *v10; // rbx
  __int64 v11; // r9
  __int64 v12; // rsi
  __int64 *v13; // rax
  __int64 *v14; // rcx
  __int64 v15; // rax
  __int64 *v16; // rbx
  __int64 v17; // r12
  _QWORD *v18; // rdi
  _QWORD *v19; // rsi
  __int64 v20; // rax
  int v21; // edx
  __int64 v22; // rdi
  int v23; // esi
  unsigned int v24; // edx
  __int64 v25; // r9
  int v26; // r10d
  unsigned int v27; // r12d
  unsigned int v28; // edx
  bool v30; // zf
  char v31; // al
  char v32; // [rsp+3h] [rbp-ADh]
  int v33; // [rsp+4h] [rbp-ACh]
  __int64 *v37; // [rsp+18h] [rbp-98h]
  __int64 v38[2]; // [rsp+20h] [rbp-90h] BYREF
  unsigned __int8 v39; // [rsp+30h] [rbp-80h]
  __int64 v40; // [rsp+40h] [rbp-70h] BYREF
  __int64 *v41; // [rsp+48h] [rbp-68h]
  __int64 v42; // [rsp+50h] [rbp-60h]
  int v43; // [rsp+58h] [rbp-58h]
  unsigned __int8 v44; // [rsp+5Ch] [rbp-54h]
  char v45; // [rsp+60h] [rbp-50h] BYREF

  v5 = (__int64)a3;
  v7 = a5;
  v8 = *(__int64 **)(a2 + 112);
  v9 = &v8[*(unsigned int *)(a2 + 120)];
  v40 = 0;
  v41 = (__int64 *)&v45;
  v42 = 4;
  v43 = 0;
  v44 = 1;
  if ( v8 != v9 )
  {
    v10 = v8;
    v11 = 1;
    while ( 1 )
    {
      while ( 1 )
      {
        v12 = *v10;
        if ( (_BYTE)v11 )
          break;
LABEL_28:
        ++v10;
        sub_C8CC70((__int64)&v40, v12, (__int64)a3, a4, a5, v11);
        v11 = v44;
        if ( v9 == v10 )
          goto LABEL_9;
      }
      v13 = v41;
      a3 = &v41[HIDWORD(v42)];
      if ( v41 == a3 )
      {
LABEL_30:
        if ( HIDWORD(v42) >= (unsigned int)v42 )
          goto LABEL_28;
        ++v10;
        ++HIDWORD(v42);
        *a3 = v12;
        v11 = v44;
        ++v40;
        if ( v9 == v10 )
          goto LABEL_9;
      }
      else
      {
        while ( v12 != *v13 )
        {
          if ( a3 == ++v13 )
            goto LABEL_30;
        }
        if ( v9 == ++v10 )
        {
LABEL_9:
          v7 = a5;
          break;
        }
      }
    }
  }
  v14 = *(__int64 **)(v5 + 64);
  v37 = &v14[*(unsigned int *)(v5 + 72)];
  if ( v14 == v37 )
  {
    v27 = 0;
    goto LABEL_40;
  }
  v15 = v7;
  v32 = 1;
  v16 = *(__int64 **)(v5 + 64);
  v33 = 0;
  v17 = v15;
  do
  {
    while ( 1 )
    {
      v20 = *v16;
      v38[0] = v20;
      if ( a2 == v20 )
        goto LABEL_18;
      if ( !v17 )
        goto LABEL_13;
      if ( *(_DWORD *)(v17 + 16) )
        break;
      v18 = *(_QWORD **)(v17 + 32);
      v19 = &v18[*(unsigned int *)(v17 + 40)];
      if ( v19 != sub_3510810(v18, (__int64)v19, v38) )
        goto LABEL_13;
LABEL_18:
      if ( v37 == ++v16 )
        goto LABEL_36;
    }
    v21 = *(_DWORD *)(v17 + 24);
    v22 = *(_QWORD *)(v17 + 8);
    if ( !v21 )
      goto LABEL_18;
    v23 = v21 - 1;
    v24 = (v21 - 1) & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
    v25 = *(_QWORD *)(v22 + 8LL * v24);
    if ( v20 != v25 )
    {
      v26 = 1;
      while ( v25 != -4096 )
      {
        v24 = v23 & (v26 + v24);
        v25 = *(_QWORD *)(v22 + 8LL * v24);
        if ( v20 == v25 )
          goto LABEL_13;
        ++v26;
      }
      goto LABEL_18;
    }
LABEL_13:
    if ( *sub_3515040(a1 + 888, v38) == a4 && *(_DWORD *)(v5 + 120) )
      goto LABEL_18;
    if ( !(unsigned __int8)sub_2FD7360((__int64 *)(a1 + 600), v5, v38[0]) )
    {
      if ( (unsigned int)(HIDWORD(v42) - v43) > 1 && HIDWORD(v42) - v43 == *(_DWORD *)(v38[0] + 120) )
      {
        v30 = (unsigned __int8)sub_3512080(v38[0], (__int64)&v40) == 0;
        v31 = v32;
        if ( v30 )
          v31 = 0;
        v32 = v31;
      }
      else
      {
        v32 = 0;
      }
      goto LABEL_18;
    }
    ++v33;
    ++v16;
  }
  while ( v37 != v16 );
LABEL_36:
  v27 = 0;
  if ( v33 )
  {
    sub_B2EE70((__int64)v38, **(_QWORD **)(a1 + 520), 0);
    v27 = v39;
    if ( !v39 )
    {
      v28 = *(_DWORD *)(v5 + 120);
      v27 = 1;
      if ( v28 )
        LOBYTE(v27) = v32 & (v28 >= v33 + 1);
    }
  }
LABEL_40:
  if ( !v44 )
    _libc_free((unsigned __int64)v41);
  return v27;
}
