// Function: sub_378DF80
// Address: 0x378df80
//
__int64 __fastcall sub_378DF80(unsigned __int64 **a1, __m128i a2)
{
  __int16 *v3; // rax
  __int16 v4; // dx
  __int64 v5; // rcx
  __int64 *v6; // rax
  __int64 v7; // r9
  __int64 v8; // rax
  __int64 (__fastcall *v9)(__int64, __int64, unsigned int, __int64); // r8
  __int64 v10; // rsi
  int v11; // eax
  unsigned __int64 *v12; // r13
  unsigned __int64 v13; // r12
  __int64 v14; // rdx
  unsigned __int8 v15; // al
  unsigned __int16 v16; // ax
  __int64 result; // rax
  __int64 v18; // rdx
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  _QWORD *v22; // r12
  unsigned int v23; // eax
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  unsigned __int8 *v27; // rax
  __int64 v28; // r8
  unsigned int v29; // r9d
  int v30; // edx
  int v31; // edi
  unsigned __int8 *v32; // rdx
  unsigned __int64 *v33; // rax
  unsigned __int64 v34; // rsi
  __int64 v35; // [rsp+20h] [rbp-50h] BYREF
  __int64 v36; // [rsp+28h] [rbp-48h]
  int v37; // [rsp+30h] [rbp-40h] BYREF
  __int64 v38; // [rsp+38h] [rbp-38h]
  __int64 v39; // [rsp+40h] [rbp-30h]

  v3 = *(__int16 **)(**a1 + 48);
  v4 = *v3;
  v5 = *((_QWORD *)v3 + 1);
  v6 = (__int64 *)a1[1];
  v36 = v5;
  LOWORD(v35) = v4;
  v7 = *v6;
  v8 = v6[1];
  v9 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v7 + 592LL);
  if ( v9 == sub_2D56A50 )
  {
    v10 = v7;
    sub_2FE6CC0((__int64)&v37, v7, *(_QWORD *)(v8 + 64), v35, v36);
    LOWORD(v11) = v38;
    LOWORD(v37) = v38;
    v38 = v39;
  }
  else
  {
    v10 = *(_QWORD *)(v8 + 64);
    v11 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v9)(v7, v10, (unsigned int)v35);
    v37 = v11;
    v38 = v18;
  }
  v12 = a1[1];
  v13 = *v12;
  v14 = *(unsigned int *)(**a1 + 24);
  if ( (_WORD)v11 == 1 || (_WORD)v11 && *(_QWORD *)(v13 + 8LL * (unsigned __int16)v11 + 112) )
  {
    if ( (unsigned int)v14 > 0x1F3 )
      return 0;
    v15 = *(_BYTE *)((unsigned int)v14 + v13 + 500LL * (unsigned __int16)v11 + 6414);
    if ( v15 <= 1u || v15 == 4 )
      return 0;
  }
  v16 = v35;
  if ( (_WORD)v35 )
  {
    if ( (unsigned __int16)(v35 - 17) > 0xD3u )
    {
LABEL_11:
      if ( *(_QWORD *)(v13 + 8LL * v16 + 112)
        && ((unsigned int)v14 > 0x1F3 || *(_BYTE *)(v14 + 500LL * v16 + v13 + 6414) != 2) )
      {
        return 0;
      }
      goto LABEL_22;
    }
    v16 = word_4456580[(unsigned __int16)v35 - 1];
  }
  else
  {
    if ( !sub_30070B0((__int64)&v35) )
      goto LABEL_18;
    v16 = sub_3009970((__int64)&v35, v10, v19, v20, v21);
    v14 = *(unsigned int *)(**a1 + 24);
  }
  if ( v16 )
    goto LABEL_11;
LABEL_22:
  v12 = a1[1];
LABEL_18:
  v22 = (_QWORD *)v12[1];
  v23 = sub_3281500(&v37, v10);
  v27 = sub_3412A00(v22, **a1, v23, v24, v25, v26, a2);
  v31 = v30;
  v32 = v27;
  v33 = a1[2];
  *v33 = (unsigned __int64)v32;
  *((_DWORD *)v33 + 2) = v31;
  v34 = **a1;
  result = 1;
  if ( *(_DWORD *)(v34 + 68) > 1u )
  {
    sub_378DDD0((__int64 *)a1[1], v34, *a1[2], *(_DWORD *)a1[3], a2, v28, v29);
    return 1;
  }
  return result;
}
