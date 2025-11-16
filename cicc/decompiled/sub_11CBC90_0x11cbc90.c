// Function: sub_11CBC90
// Address: 0x11cbc90
//
__int64 __fastcall sub_11CBC90(__int64 a1, __int64 a2, __int64 a3, __int64 *a4, unsigned int a5, unsigned __int8 a6)
{
  __int64 *v10; // r13
  bool v11; // r8
  __int64 result; // rax
  __int64 v13; // r9
  __int64 v14; // rsi
  __int64 *v15; // r14
  __int64 v16; // rax
  __int64 v17; // rcx
  __int64 v18; // rdx
  unsigned __int64 v19; // rax
  unsigned __int64 v20; // r9
  __int64 v21; // rdx
  __int64 v22; // r14
  _QWORD *v23; // rdi
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rdi
  unsigned int v27; // ecx
  int *v28; // rdx
  int v29; // esi
  int v30; // edx
  int v31; // r9d
  unsigned __int64 v32; // [rsp+10h] [rbp-B0h]
  unsigned __int64 v33; // [rsp+20h] [rbp-A0h]
  unsigned __int64 v34; // [rsp+20h] [rbp-A0h]
  char *v36; // [rsp+30h] [rbp-90h]
  _QWORD v38[4]; // [rsp+40h] [rbp-80h] BYREF
  const char *v39; // [rsp+60h] [rbp-60h] BYREF
  __int64 v40; // [rsp+68h] [rbp-58h]
  _QWORD v41[2]; // [rsp+70h] [rbp-50h] BYREF
  __int64 v42; // [rsp+80h] [rbp-40h]

  v10 = (__int64 *)sub_AA4B30(*(_QWORD *)(a3 + 48));
  v11 = sub_11C99B0(v10, a4, a5);
  result = 0;
  if ( !v11 )
    return result;
  v13 = a4[((unsigned __int64)a5 >> 6) + 1] & (1LL << a5);
  if ( v13 )
  {
    v36 = 0;
    v13 = 0;
    goto LABEL_7;
  }
  v14 = *a4;
  if ( (((int)*(unsigned __int8 *)(*a4 + (a5 >> 2)) >> (2 * (a5 & 3))) & 3) == 0 )
  {
    v36 = 0;
    goto LABEL_7;
  }
  if ( (((int)*(unsigned __int8 *)(*a4 + (a5 >> 2)) >> (2 * (a5 & 3))) & 3) != 3 )
  {
    v25 = *(unsigned int *)(v14 + 160);
    v26 = *(_QWORD *)(v14 + 144);
    if ( (_DWORD)v25 )
    {
      v27 = (v25 - 1) & (37 * a5);
      v28 = (int *)(v26 + 40LL * v27);
      v29 = *v28;
      if ( a5 == *v28 )
      {
LABEL_14:
        v13 = *((_QWORD *)v28 + 2);
        v36 = (char *)*((_QWORD *)v28 + 1);
        goto LABEL_7;
      }
      v30 = 1;
      while ( v29 != -1 )
      {
        v31 = v30 + 1;
        v27 = (v25 - 1) & (v30 + v27);
        v28 = (int *)(v26 + 40LL * v27);
        v29 = *v28;
        if ( a5 == *v28 )
          goto LABEL_14;
        v30 = v31;
      }
    }
    v28 = (int *)(v26 + 40 * v25);
    goto LABEL_14;
  }
  v13 = qword_4977328[2 * a5];
  v36 = (&off_4977320)[2 * a5];
LABEL_7:
  v33 = v13;
  v39 = (const char *)sub_BCE3C0(*(__int64 **)(a3 + 72), 0);
  v40 = *(_QWORD *)(a1 + 8);
  v15 = sub_BD0B90((_QWORD *)*v10, &v39, 2, 0);
  v16 = sub_BCB2B0(*(_QWORD **)(a3 + 72));
  v17 = *(_QWORD *)(a1 + 8);
  v42 = v16;
  v41[0] = v17;
  v18 = *(_QWORD *)(a2 + 8);
  v39 = (const char *)v41;
  v41[1] = v18;
  v40 = 0x300000003LL;
  v19 = sub_BCF480(v15, v41, 3, 0);
  v32 = v33;
  v34 = sub_BA8C10((__int64)v10, (__int64)v36, v33, v19, 0);
  v20 = v32;
  v22 = v21;
  if ( v39 != (const char *)v41 )
  {
    _libc_free(v39, v36);
    v20 = v32;
  }
  sub_11C9500((__int64)v10, (__int64)v36, v20, a4);
  v23 = *(_QWORD **)(a3 + 72);
  v39 = "sized_ptr";
  LOWORD(v42) = 259;
  v38[0] = a1;
  v38[1] = a2;
  v24 = sub_BCB2B0(v23);
  v38[2] = sub_ACD640(v24, a6, 0);
  result = sub_921880((unsigned int **)a3, v34, v22, (int)v38, 3, (__int64)&v39, 0);
  if ( !*(_BYTE *)v22 )
    *(_WORD *)(result + 2) = *(_WORD *)(result + 2) & 0xF003 | (4 * ((*(_WORD *)(v22 + 2) >> 4) & 0x3FF));
  return result;
}
