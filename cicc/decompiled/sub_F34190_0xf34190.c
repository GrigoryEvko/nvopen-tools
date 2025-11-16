// Function: sub_F34190
// Address: 0xf34190
//
__int64 *__fastcall sub_F34190(__int64 *a1, __int64 a2, __int64 a3, unsigned __int8 a4)
{
  __int64 *result; // rax
  __int64 v6; // r12
  __int64 v7; // rdx
  unsigned __int64 v8; // r13
  __int64 v9; // r14
  int v10; // eax
  unsigned int v11; // r15d
  __int64 v12; // r13
  __int64 *v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 *v17; // rax
  __int64 v18; // rax
  unsigned __int64 v19; // rdx
  _QWORD *v20; // rdi
  __int64 v21; // r13
  __int64 v22; // rsi
  _QWORD *v23; // rdi
  char v24; // dl
  __int64 v25; // rdx
  __int64 v26; // rax
  __int64 *v27; // rdx
  const void *v28; // [rsp+0h] [rbp-C0h]
  __int64 *v29; // [rsp+10h] [rbp-B0h]
  __int64 v30; // [rsp+18h] [rbp-A8h]
  __int64 v31; // [rsp+28h] [rbp-98h]
  __int64 *v32; // [rsp+30h] [rbp-90h]
  int v34; // [rsp+3Ch] [rbp-84h]
  __int64 v35; // [rsp+40h] [rbp-80h] BYREF
  __int64 v36; // [rsp+48h] [rbp-78h]
  __int64 v37; // [rsp+50h] [rbp-70h] BYREF
  __int64 *v38; // [rsp+58h] [rbp-68h]
  __int64 v39; // [rsp+60h] [rbp-60h]
  int v40; // [rsp+68h] [rbp-58h]
  char v41; // [rsp+6Ch] [rbp-54h]
  char v42; // [rsp+70h] [rbp-50h] BYREF

  result = &a1[a2];
  v28 = (const void *)(a3 + 16);
  v29 = result;
  v32 = a1;
  if ( a1 == result )
    return result;
  do
  {
    v37 = 0;
    v39 = 4;
    v6 = *v32;
    v40 = 0;
    v41 = 1;
    v7 = *(_QWORD *)(v6 + 48);
    v38 = (__int64 *)&v42;
    v31 = v6 + 48;
    v8 = v7 & 0xFFFFFFFFFFFFFFF8LL;
    if ( v6 + 48 == (v7 & 0xFFFFFFFFFFFFFFF8LL) )
      goto LABEL_21;
    if ( !v8 )
      BUG();
    v9 = v8 - 24;
    if ( (unsigned int)*(unsigned __int8 *)(v8 - 24) - 30 > 0xA
      || (v30 = v7, v10 = sub_B46E30(v8 - 24), v7 = v30, (v34 = v10) == 0) )
    {
      do
      {
LABEL_15:
        if ( !v8 )
          BUG();
        if ( *(_QWORD *)(v8 - 8) )
        {
          v18 = sub_ACADE0(*(__int64 ***)(v8 - 16));
          sub_BD84D0(v8 - 24, v18);
          v7 = *(_QWORD *)(v6 + 48);
        }
        v19 = v7 & 0xFFFFFFFFFFFFFFF8LL;
        v20 = (_QWORD *)(v19 - 24);
        if ( !v19 )
          v20 = 0;
        sub_B43D60(v20);
        v7 = *(_QWORD *)(v6 + 48);
        v8 = v7 & 0xFFFFFFFFFFFFFFF8LL;
      }
      while ( (v7 & 0xFFFFFFFFFFFFFFF8LL) != v31 );
      goto LABEL_21;
    }
    v11 = 0;
    do
    {
      while ( 1 )
      {
        v12 = sub_B46EC0(v9, v11);
        sub_AA5980(v12, v6, a4);
        if ( !a3 )
          goto LABEL_13;
        if ( v41 )
        {
          v17 = v38;
          v13 = &v38[HIDWORD(v39)];
          if ( v38 != v13 )
          {
            while ( v12 != *v17 )
            {
              if ( v13 == ++v17 )
                goto LABEL_32;
            }
            goto LABEL_13;
          }
LABEL_32:
          if ( HIDWORD(v39) < (unsigned int)v39 )
            break;
        }
        sub_C8CC70((__int64)&v37, v12, (__int64)v13, v14, v15, v16);
        if ( v24 )
          goto LABEL_28;
LABEL_13:
        if ( v34 == ++v11 )
          goto LABEL_14;
      }
      ++HIDWORD(v39);
      *v13 = v12;
      ++v37;
LABEL_28:
      v25 = *(unsigned int *)(a3 + 8);
      v26 = v12 | 4;
      if ( v25 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
      {
        sub_C8D5F0(a3, v28, v25 + 1, 0x10u, v15, v25 + 1);
        v25 = *(unsigned int *)(a3 + 8);
        v26 = v12 | 4;
      }
      v27 = (__int64 *)(*(_QWORD *)a3 + 16 * v25);
      ++v11;
      *v27 = v6;
      v27[1] = v26;
      ++*(_DWORD *)(a3 + 8);
    }
    while ( v34 != v11 );
LABEL_14:
    v7 = *(_QWORD *)(v6 + 48);
    v8 = v7 & 0xFFFFFFFFFFFFFFF8LL;
    if ( v31 != (v7 & 0xFFFFFFFFFFFFFFF8LL) )
      goto LABEL_15;
LABEL_21:
    v21 = sub_AA48A0(v6);
    sub_B43C20((__int64)&v35, v6);
    v22 = unk_3F148B8;
    v23 = sub_BD2C40(72, unk_3F148B8);
    if ( v23 )
    {
      v22 = v21;
      sub_B4C8A0((__int64)v23, v21, v35, v36);
    }
    if ( !v41 )
      _libc_free(v38, v22);
    result = ++v32;
  }
  while ( v29 != v32 );
  return result;
}
