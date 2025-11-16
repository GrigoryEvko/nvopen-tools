// Function: sub_DFD800
// Address: 0xdfd800
//
__int64 __fastcall sub_DFD800(
        __int64 a1,
        unsigned int a2,
        __int64 a3,
        unsigned int a4,
        __int64 a5,
        __int64 a6,
        _QWORD *a7,
        __int64 a8,
        __int64 a9,
        __int64 *a10)
{
  __int64 v10; // r10
  __int64 v15; // rdi
  __int64 (__fastcall *v16)(__int64, unsigned int, __int64, int, __int64, __int64, _QWORD *, __int64); // rax
  __int64 result; // rax
  _QWORD *v18; // r8
  _QWORD *v19; // r9
  int v20; // edx
  char v21; // al
  __int64 v22; // rdi
  size_t v23; // r11
  char **v24; // rax
  char *v25; // rsi
  __int64 v26; // rdx
  __int64 v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // rsi
  int *v30; // rdx
  int v31; // r8d
  int v32; // edx
  int v33; // r8d
  __int64 v34; // rdx
  __int64 v35; // [rsp+0h] [rbp-80h]
  __int64 v36; // [rsp+8h] [rbp-78h]
  __int64 v37; // [rsp+20h] [rbp-60h]
  size_t v38; // [rsp+20h] [rbp-60h]
  int v39; // [rsp+20h] [rbp-60h]
  unsigned int v41; // [rsp+28h] [rbp-58h]
  unsigned int v42; // [rsp+3Ch] [rbp-44h] BYREF
  _QWORD v43[8]; // [rsp+40h] [rbp-40h] BYREF

  v10 = a5;
  if ( a10 && a2 == 24 )
  {
    if ( (unsigned __int8)(*(_BYTE *)(a3 + 8) - 17) > 1u )
      goto LABEL_4;
    v37 = a6;
    v21 = sub_97F7A0(*a10, 24, **(_QWORD **)(a3 + 16), (int *)&v42);
    v10 = a5;
    a6 = v37;
    if ( !v21 )
      goto LABEL_4;
    BYTE4(v43[0]) = *(_BYTE *)(a3 + 8) == 18;
    LODWORD(v43[0]) = *(_DWORD *)(a3 + 32);
    v22 = *a10;
    v23 = a10[((unsigned __int64)v42 >> 6) + 1] & (1LL << v42);
    if ( v23 )
    {
      v23 = 0;
      v25 = 0;
      goto LABEL_25;
    }
    if ( (((int)*(unsigned __int8 *)(v22 + (v42 >> 2)) >> (2 * (v42 & 3))) & 3) == 0 )
    {
      v25 = 0;
      goto LABEL_25;
    }
    if ( (((int)*(unsigned __int8 *)(v22 + (v42 >> 2)) >> (2 * (v42 & 3))) & 3) == 3 )
    {
      v24 = &(&off_4977320)[2 * v42];
      v25 = *v24;
      v23 = (size_t)v24[1];
LABEL_25:
      v35 = a6;
      v36 = v10;
      v38 = v23;
      sub_97FA10(v22, v25, v23, (__int64)v43, 0);
      if ( v26 || (sub_97FA10(v22, v25, v38, (__int64)v43, 1), v10 = v36, a6 = v35, v27) )
      {
        v43[0] = a3;
        v43[1] = a3;
        return sub_DFD7B0(a1);
      }
LABEL_4:
      v15 = *(_QWORD *)a1;
      v16 = *(__int64 (__fastcall **)(__int64, unsigned int, __int64, int, __int64, __int64, _QWORD *, __int64))(**(_QWORD **)a1 + 1176LL);
      if ( v16 == sub_DF7260 )
        return 4;
      return ((__int64 (__fastcall *)(__int64, _QWORD, __int64, _QWORD, __int64, __int64, _QWORD *, __int64, __int64))v16)(
               v15,
               a2,
               a3,
               a4,
               v10,
               a6,
               a7,
               a8,
               a9);
    }
    v28 = *(unsigned int *)(v22 + 160);
    v29 = *(_QWORD *)(v22 + 144);
    if ( (_DWORD)v28 )
    {
      v41 = (v28 - 1) & (37 * v42);
      v30 = (int *)(v29 + 40LL * v41);
      v31 = *v30;
      if ( v42 == *v30 )
      {
LABEL_34:
        v25 = (char *)*((_QWORD *)v30 + 1);
        v23 = *((_QWORD *)v30 + 2);
        goto LABEL_25;
      }
      v32 = 1;
      while ( v31 != -1 )
      {
        v33 = v32 + 1;
        v34 = ((_DWORD)v28 - 1) & (v41 + v32);
        v39 = v33;
        v41 = v34;
        v30 = (int *)(v29 + 40 * v34);
        v31 = *v30;
        if ( v42 == *v30 )
          goto LABEL_34;
        v32 = v39;
      }
    }
    v30 = (int *)(v29 + 40 * v28);
    goto LABEL_34;
  }
  v15 = *(_QWORD *)a1;
  v16 = *(__int64 (__fastcall **)(__int64, unsigned int, __int64, int, __int64, __int64, _QWORD *, __int64))(**(_QWORD **)a1 + 1176LL);
  if ( v16 != sub_DF7260 )
    return ((__int64 (__fastcall *)(__int64, _QWORD, __int64, _QWORD, __int64, __int64, _QWORD *, __int64, __int64))v16)(
             v15,
             a2,
             a3,
             a4,
             v10,
             a6,
             a7,
             a8,
             a9);
  if ( a2 <= 0x18 )
  {
    if ( a2 > 0x12 )
      return 4;
  }
  else if ( a2 - 28 <= 1 )
  {
    v18 = sub_DF7050(a7, (__int64)&a7[a8]);
    result = 0;
    if ( v19 != v18 )
      return result;
    result = 1;
    if ( a4 != 1 )
      return result;
    goto LABEL_12;
  }
  result = 1;
  if ( a4 != 1 )
    return result;
LABEL_12:
  v20 = *(unsigned __int8 *)(a3 + 8);
  if ( (unsigned int)(v20 - 17) <= 1 )
    LOBYTE(v20) = *(_BYTE *)(**(_QWORD **)(a3 + 16) + 8LL);
  result = 3;
  if ( (unsigned __int8)v20 > 3u && (_BYTE)v20 != 5 )
    return 2LL * ((v20 & 0xFD) == 4) + 1;
  return result;
}
