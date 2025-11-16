// Function: sub_11CB9C0
// Address: 0x11cb9c0
//
__int64 __fastcall sub_11CB9C0(__int64 a1, __int64 a2, __int64 *a3, unsigned int a4, unsigned __int8 a5)
{
  __int64 *v9; // r13
  bool v10; // r8
  __int64 result; // rax
  __int64 v12; // r9
  __int64 v13; // rsi
  __int64 *v14; // r14
  __int64 v15; // rax
  __int64 v16; // rdx
  unsigned __int64 v17; // rax
  unsigned __int64 v18; // r9
  __int64 v19; // rdx
  __int64 v20; // r14
  _QWORD *v21; // rdi
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rdi
  unsigned int v25; // ecx
  int *v26; // rdx
  int v27; // esi
  int v28; // edx
  int v29; // r9d
  unsigned __int64 v30; // [rsp+8h] [rbp-98h]
  unsigned __int64 v31; // [rsp+18h] [rbp-88h]
  unsigned __int64 v32; // [rsp+18h] [rbp-88h]
  char *v34; // [rsp+28h] [rbp-78h]
  _QWORD v35[2]; // [rsp+30h] [rbp-70h] BYREF
  const char *v36; // [rsp+40h] [rbp-60h] BYREF
  __int64 v37; // [rsp+48h] [rbp-58h]
  _QWORD v38[2]; // [rsp+50h] [rbp-50h] BYREF
  char v39; // [rsp+60h] [rbp-40h]
  char v40; // [rsp+61h] [rbp-3Fh]

  v9 = (__int64 *)sub_AA4B30(*(_QWORD *)(a2 + 48));
  v10 = sub_11C99B0(v9, a3, a4);
  result = 0;
  if ( !v10 )
    return result;
  v12 = a3[((unsigned __int64)a4 >> 6) + 1] & (1LL << a4);
  if ( v12 )
  {
    v34 = 0;
    v12 = 0;
    goto LABEL_7;
  }
  v13 = *a3;
  if ( (((int)*(unsigned __int8 *)(*a3 + (a4 >> 2)) >> (2 * (a4 & 3))) & 3) == 0 )
  {
    v34 = 0;
    goto LABEL_7;
  }
  if ( (((int)*(unsigned __int8 *)(*a3 + (a4 >> 2)) >> (2 * (a4 & 3))) & 3) != 3 )
  {
    v23 = *(unsigned int *)(v13 + 160);
    v24 = *(_QWORD *)(v13 + 144);
    if ( (_DWORD)v23 )
    {
      v25 = (v23 - 1) & (37 * a4);
      v26 = (int *)(v24 + 40LL * v25);
      v27 = *v26;
      if ( a4 == *v26 )
      {
LABEL_14:
        v12 = *((_QWORD *)v26 + 2);
        v34 = (char *)*((_QWORD *)v26 + 1);
        goto LABEL_7;
      }
      v28 = 1;
      while ( v27 != -1 )
      {
        v29 = v28 + 1;
        v25 = (v23 - 1) & (v28 + v25);
        v26 = (int *)(v24 + 40LL * v25);
        v27 = *v26;
        if ( a4 == *v26 )
          goto LABEL_14;
        v28 = v29;
      }
    }
    v26 = (int *)(v24 + 40 * v23);
    goto LABEL_14;
  }
  v12 = qword_4977328[2 * a4];
  v34 = (&off_4977320)[2 * a4];
LABEL_7:
  v31 = v12;
  v36 = (const char *)sub_BCE3C0(*(__int64 **)(a2 + 72), 0);
  v37 = *(_QWORD *)(a1 + 8);
  v14 = sub_BD0B90((_QWORD *)*v9, &v36, 2, 0);
  v15 = sub_BCB2B0(*(_QWORD **)(a2 + 72));
  v16 = *(_QWORD *)(a1 + 8);
  v38[1] = v15;
  v38[0] = v16;
  v36 = (const char *)v38;
  v37 = 0x200000002LL;
  v17 = sub_BCF480(v14, v38, 2, 0);
  v30 = v31;
  v32 = sub_BA8C10((__int64)v9, (__int64)v34, v31, v17, 0);
  v18 = v30;
  v20 = v19;
  if ( v36 != (const char *)v38 )
  {
    _libc_free(v36, v34);
    v18 = v30;
  }
  sub_11C9500((__int64)v9, (__int64)v34, v18, a3);
  v21 = *(_QWORD **)(a2 + 72);
  v40 = 1;
  v36 = "sized_ptr";
  v39 = 3;
  v35[0] = a1;
  v22 = sub_BCB2B0(v21);
  v35[1] = sub_ACD640(v22, a5, 0);
  result = sub_921880((unsigned int **)a2, v32, v20, (int)v35, 2, (__int64)&v36, 0);
  if ( !*(_BYTE *)v20 )
    *(_WORD *)(result + 2) = *(_WORD *)(result + 2) & 0xF003 | (4 * ((*(_WORD *)(v20 + 2) >> 4) & 0x3FF));
  return result;
}
