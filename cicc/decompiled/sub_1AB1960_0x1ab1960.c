// Function: sub_1AB1960
// Address: 0x1ab1960
//
__int64 __fastcall sub_1AB1960(__int64 a1, __int64 a2, __int64 a3, __int64 *a4)
{
  __int64 v4; // r14
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // r14
  int v12; // eax
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rdi
  __int64 *v16; // rax
  __int64 v17; // rax
  __int64 v18; // rbx
  __int64 v19; // rax
  __int64 v21; // rax
  __int64 v22; // rsi
  unsigned int v23; // edi
  int *v24; // rcx
  int v25; // edx
  int v26; // ecx
  int v27; // r9d
  __int64 v28; // [rsp+0h] [rbp-80h]
  __int64 v29; // [rsp+8h] [rbp-78h]
  __int64 v30; // [rsp+18h] [rbp-68h] BYREF
  __int64 v31; // [rsp+20h] [rbp-60h] BYREF
  __int64 v32; // [rsp+28h] [rbp-58h]
  _QWORD v33[2]; // [rsp+30h] [rbp-50h] BYREF
  __int64 v34[8]; // [rsp+40h] [rbp-40h] BYREF

  v4 = 0;
  if ( (*(_BYTE *)(*a4 + 92) & 0xC0) == 0 )
    return v4;
  v9 = sub_157EB90(*(_QWORD *)(a2 + 8));
  v10 = *a4;
  v11 = v9;
  v12 = (int)*(unsigned __int8 *)(*a4 + 92) >> 6;
  if ( !v12 )
  {
    v13 = 0;
    v14 = 0;
    goto LABEL_4;
  }
  if ( v12 == 3 )
  {
    v14 = qword_4F9B700[742];
    v13 = qword_4F9B700[743];
    goto LABEL_4;
  }
  v21 = *(unsigned int *)(v10 + 136);
  v22 = *(_QWORD *)(v10 + 120);
  if ( !(_DWORD)v21 )
    goto LABEL_12;
  v23 = ((_WORD)v21 - 1) & 0x359F;
  v24 = (int *)(v22 + 40LL * (((_WORD)v21 - 1) & 0x359F));
  v25 = *v24;
  if ( *v24 != 371 )
  {
    v26 = 1;
    while ( v25 != -1 )
    {
      v27 = v26 + 1;
      v23 = (v21 - 1) & (v26 + v23);
      v24 = (int *)(v22 + 40LL * v23);
      v25 = *v24;
      if ( *v24 == 371 )
        goto LABEL_11;
      v26 = v27;
    }
LABEL_12:
    v24 = (int *)(v22 + 40 * v21);
  }
LABEL_11:
  v14 = *((_QWORD *)v24 + 1);
  v13 = *((_QWORD *)v24 + 2);
LABEL_4:
  v15 = *(_QWORD *)(a2 + 8);
  v31 = v14;
  v32 = v13;
  v28 = sub_157E9C0(v15);
  v29 = sub_16471D0(*(_QWORD **)(a2 + 24), 0);
  v16 = (__int64 *)sub_15A9620(a3, v28, 0);
  v34[0] = v29;
  v33[1] = 0x100000001LL;
  v17 = sub_1644EA0(v16, v34, 1, 0);
  v18 = sub_1632080(v11, v31, v32, v17, 0);
  sub_1AB1740(v11, v31, v32, a4);
  LOWORD(v34[0]) = 261;
  v33[0] = &v31;
  v30 = sub_1AB1800(a1, (__int64 *)a2);
  v4 = sub_1285290((__int64 *)a2, *(_QWORD *)(*(_QWORD *)v18 + 24LL), v18, (int)&v30, 1, (__int64)v33, 0);
  v19 = sub_1649C60(v18);
  if ( !*(_BYTE *)(v19 + 16) )
    *(_WORD *)(v4 + 18) = *(_WORD *)(v4 + 18) & 0x8000 | *(_WORD *)(v4 + 18) & 3 | (*(_WORD *)(v19 + 18) >> 2) & 0xFFC;
  return v4;
}
