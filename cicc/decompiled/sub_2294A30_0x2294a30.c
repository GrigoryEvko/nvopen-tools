// Function: sub_2294A30
// Address: 0x2294a30
//
char __fastcall sub_2294A30(__int64 a1, __int64 a2, __int64 a3, int *a4, __int64 a5, __int64 a6, __int64 *a7)
{
  __int16 v10; // ax
  __int64 v11; // rsi
  __int64 *v12; // rdx
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // rax
  char *v17; // r15
  int v18; // eax
  char v19; // al
  __int64 v21; // r15
  int v22; // eax
  __int64 v23; // r15
  int v24; // eax
  __int64 v25; // [rsp+8h] [rbp-68h]
  __int64 v26; // [rsp+10h] [rbp-60h]
  __int64 v27; // [rsp+20h] [rbp-50h]
  __int64 v28; // [rsp+20h] [rbp-50h]
  __int64 v29; // [rsp+20h] [rbp-50h]
  __int64 *v30; // [rsp+28h] [rbp-48h]
  char *v31; // [rsp+28h] [rbp-48h]
  char *v32; // [rsp+28h] [rbp-48h]

  v10 = *(_WORD *)(a3 + 24);
  if ( *(_WORD *)(a2 + 24) == 8 )
  {
    v11 = *(_QWORD *)(a1 + 8);
    v12 = *(__int64 **)(a2 + 32);
    if ( v10 == 8 )
    {
      v26 = *v12;
      v25 = **(_QWORD **)(a3 + 32);
      v30 = (__int64 *)sub_D33D80((_QWORD *)a2, v11, (__int64)v12, (__int64)a4, a5);
      v16 = sub_D33D80((_QWORD *)a3, *(_QWORD *)(a1 + 8), v13, v14, v15);
      v17 = *(char **)(a2 + 48);
      v27 = v16;
      v18 = sub_228D710(a1, (_QWORD **)v17);
      *a4 = v18;
      if ( v30 == (__int64 *)v27 )
      {
        v19 = sub_228ECD0(a1, (__int64)v30, v26, v25, v17, v18, a5, a6);
      }
      else if ( v30 == sub_DCAF50(*(__int64 **)(a1 + 8), v27, 0) )
      {
        v19 = sub_228F1E0(a1, (__int64)v30, v26, v25, v17, *a4, a5, a6, a7);
      }
      else
      {
        v19 = sub_22936D0(a1, (__int64)v30, v27, v26, v25, v17, *a4, a5, a6);
      }
      if ( !v19 && !(unsigned __int8)sub_228FE90(a1, a2, a3, a5) )
        return sub_228FB70(a1, (__int64)v30, v27, v26, v25, v17, v17);
      return 1;
    }
    v28 = *v12;
    v21 = sub_D33D80((_QWORD *)a2, v11, *v12, (__int64)a4, a5);
    v31 = *(char **)(a2 + 48);
    v22 = sub_228D710(a1, (_QWORD **)v31);
    *a4 = v22;
    if ( (unsigned __int8)sub_228F910(a1, v21, v28, a3, v31, v22, a5, a6) )
      return 1;
  }
  else
  {
    if ( v10 != 8 )
      BUG();
    v29 = **(_QWORD **)(a3 + 32);
    v23 = sub_D33D80((_QWORD *)a3, *(_QWORD *)(a1 + 8), a3, v29, a5);
    v32 = *(char **)(a3 + 48);
    v24 = sub_228D730(a1, (_QWORD **)v32);
    *a4 = v24;
    if ( (unsigned __int8)sub_228F6B0(a1, v23, a2, v29, v32, v24, a5, a6) )
      return 1;
  }
  return sub_228FE90(a1, a2, a3, a5);
}
