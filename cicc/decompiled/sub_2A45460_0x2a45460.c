// Function: sub_2A45460
// Address: 0x2a45460
//
__int64 __fastcall sub_2A45460(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  bool v7; // zf
  __int64 v8; // r15
  char v9; // r14
  __int64 v10; // r12
  __int64 v11; // rsi
  __int64 *v12; // rax
  _QWORD *v13; // r13
  _QWORD *v14; // r12
  __int64 v15; // rax
  char v16; // dl
  __int64 *v17; // rax
  __int64 *v18; // r13
  _QWORD *v19; // rdi
  __int64 *v20; // r12
  _QWORD *v21; // r13
  _QWORD *v22; // r12
  __int64 v23; // rax
  unsigned __int64 *v24; // r12
  __int64 result; // rax
  unsigned __int64 *i; // rbx
  unsigned __int64 *v27; // rax
  unsigned __int64 v28; // rcx
  __int64 *v29; // rax
  __int64 v30; // [rsp+8h] [rbp-158h]
  __int64 v31; // [rsp+10h] [rbp-150h] BYREF
  __int64 *v32; // [rsp+18h] [rbp-148h]
  __int64 v33; // [rsp+20h] [rbp-140h]
  int v34; // [rsp+28h] [rbp-138h]
  char v35; // [rsp+2Ch] [rbp-134h]
  char v36; // [rsp+30h] [rbp-130h] BYREF

  v32 = (__int64 *)&v36;
  v7 = *(_QWORD *)(a1 + 592) == 0;
  v31 = 0;
  v33 = 32;
  v34 = 0;
  v35 = 1;
  v30 = a1 + 560;
  if ( v7 )
  {
    v8 = *(_QWORD *)(a1 + 56);
    v9 = 1;
    v10 = v8 + 24LL * *(unsigned int *)(a1 + 64);
  }
  else
  {
    v8 = *(_QWORD *)(a1 + 576);
    v10 = a1 + 560;
    v9 = 0;
  }
  if ( v9 )
    goto LABEL_12;
LABEL_4:
  if ( v10 == v8 )
    goto LABEL_18;
  v11 = *(_QWORD *)(v8 + 48);
  if ( v35 )
  {
LABEL_6:
    v12 = v32;
    a4 = HIDWORD(v33);
    a3 = &v32[HIDWORD(v33)];
    if ( v32 == a3 )
    {
LABEL_16:
      if ( HIDWORD(v33) >= (unsigned int)v33 )
        goto LABEL_14;
      a4 = (unsigned int)++HIDWORD(v33);
      *a3 = v11;
      ++v31;
    }
    else
    {
      while ( v11 != *v12 )
      {
        if ( a3 == ++v12 )
          goto LABEL_16;
      }
    }
    if ( !v9 )
      goto LABEL_15;
    goto LABEL_11;
  }
  while ( 1 )
  {
LABEL_14:
    sub_C8CC70((__int64)&v31, v11, (__int64)a3, a4, a5, a6);
    if ( !v9 )
    {
LABEL_15:
      v8 = sub_220EF30(v8);
      goto LABEL_4;
    }
LABEL_11:
    v8 += 24;
LABEL_12:
    if ( v10 == v8 )
      break;
    v11 = *(_QWORD *)(v8 + 16);
    if ( v35 )
      goto LABEL_6;
  }
LABEL_18:
  v13 = *(_QWORD **)(a1 + 56);
  v14 = &v13[3 * *(unsigned int *)(a1 + 64)];
  while ( v13 != v14 )
  {
    while ( 1 )
    {
      v15 = *(v14 - 1);
      v14 -= 3;
      if ( v15 == 0 || v15 == -4096 || v15 == -8192 )
        break;
      sub_BD60C0(v14);
      if ( v13 == v14 )
        goto LABEL_23;
    }
  }
LABEL_23:
  *(_DWORD *)(a1 + 64) = 0;
  sub_2A44860(*(_QWORD **)(a1 + 568));
  v16 = v35;
  *(_QWORD *)(a1 + 568) = 0;
  *(_QWORD *)(a1 + 592) = 0;
  *(_QWORD *)(a1 + 576) = v30;
  *(_QWORD *)(a1 + 584) = v30;
  v17 = v32;
  if ( v16 )
  {
    v18 = &v32[HIDWORD(v33)];
    if ( v32 != v18 )
      goto LABEL_25;
  }
  else
  {
    v18 = &v32[(unsigned int)v33];
    if ( v32 != v18 )
    {
LABEL_25:
      while ( 1 )
      {
        v19 = (_QWORD *)*v17;
        v20 = v17;
        if ( (unsigned __int64)*v17 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v18 == ++v17 )
          goto LABEL_27;
      }
      if ( v18 == v17 )
      {
LABEL_27:
        if ( !v16 )
          goto LABEL_44;
        goto LABEL_28;
      }
      do
      {
        sub_B2E860(v19);
        v29 = v20 + 1;
        if ( v20 + 1 == v18 )
          break;
        while ( 1 )
        {
          v19 = (_QWORD *)*v29;
          v20 = v29;
          if ( (unsigned __int64)*v29 < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( v18 == ++v29 )
            goto LABEL_43;
        }
      }
      while ( v18 != v29 );
LABEL_43:
      if ( v35 )
        goto LABEL_28;
    }
LABEL_44:
    _libc_free((unsigned __int64)v32);
  }
LABEL_28:
  sub_2A44860(*(_QWORD **)(a1 + 568));
  v21 = *(_QWORD **)(a1 + 56);
  v22 = &v21[3 * *(unsigned int *)(a1 + 64)];
  if ( v21 != v22 )
  {
    do
    {
      v23 = *(v22 - 1);
      v22 -= 3;
      if ( v23 != 0 && v23 != -4096 && v23 != -8192 )
        sub_BD60C0(v22);
    }
    while ( v21 != v22 );
    v22 = *(_QWORD **)(a1 + 56);
  }
  if ( v22 != (_QWORD *)(a1 + 72) )
    _libc_free((unsigned __int64)v22);
  v24 = (unsigned __int64 *)(a1 + 8);
  result = sub_C7D6A0(*(_QWORD *)(a1 + 32), 16LL * *(unsigned int *)(a1 + 48), 8);
  for ( i = *(unsigned __int64 **)(a1 + 16);
        v24 != i;
        result = (*(__int64 (__fastcall **)(unsigned __int64 *))(*(v27 - 1) + 8))(v27 - 1) )
  {
    v27 = i;
    i = (unsigned __int64 *)i[1];
    v28 = *v27 & 0xFFFFFFFFFFFFFFF8LL;
    *i = v28 | *i & 7;
    *(_QWORD *)(v28 + 8) = i;
    v27[1] = 0;
    *v27 &= 7u;
  }
  return result;
}
