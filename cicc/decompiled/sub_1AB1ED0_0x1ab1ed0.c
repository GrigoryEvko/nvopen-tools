// Function: sub_1AB1ED0
// Address: 0x1ab1ed0
//
__int64 __fastcall sub_1AB1ED0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 *a6)
{
  __int64 v6; // rbx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rbx
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rdi
  __int64 v17; // rax
  __int64 *v18; // rax
  __int64 v19; // rax
  __int64 v20; // r15
  __int64 v21; // rax
  __int64 v23; // rax
  __int64 v24; // rsi
  unsigned int v25; // edi
  int *v26; // rcx
  int v27; // edx
  int v28; // ecx
  int v29; // r10d
  __int64 v30; // [rsp+0h] [rbp-C0h]
  __int64 v31; // [rsp+8h] [rbp-B8h]
  __int64 v32; // [rsp+18h] [rbp-A8h]
  __int64 v35; // [rsp+30h] [rbp-90h] BYREF
  __int64 v36; // [rsp+38h] [rbp-88h]
  _QWORD v37[4]; // [rsp+40h] [rbp-80h] BYREF
  _QWORD v38[2]; // [rsp+60h] [rbp-60h] BYREF
  _QWORD v39[10]; // [rsp+70h] [rbp-50h] BYREF

  v6 = 0;
  if ( (*(_BYTE *)(*a6 + 72) & 0xC) == 0 )
    return v6;
  v11 = sub_157EB90(*(_QWORD *)(a4 + 8));
  v12 = *a6;
  v13 = v11;
  if ( (((int)*(unsigned __int8 *)(*a6 + 72) >> 2) & 3) == 0 )
  {
    v14 = 0;
    v15 = 0;
    goto LABEL_4;
  }
  if ( (((int)*(unsigned __int8 *)(*a6 + 72) >> 2) & 3) == 3 )
  {
    v15 = qword_4F9B700[578];
    v14 = qword_4F9B700[579];
    goto LABEL_4;
  }
  v23 = *(unsigned int *)(v12 + 136);
  v24 = *(_QWORD *)(v12 + 120);
  if ( !(_DWORD)v23 )
    goto LABEL_12;
  v25 = ((_WORD)v23 - 1) & 0x29C5;
  v26 = (int *)(v24 + 40LL * (((_WORD)v23 - 1) & 0x29C5));
  v27 = *v26;
  if ( *v26 != 289 )
  {
    v28 = 1;
    while ( v27 != -1 )
    {
      v29 = v28 + 1;
      v25 = (v23 - 1) & (v28 + v25);
      v26 = (int *)(v24 + 40LL * v25);
      v27 = *v26;
      if ( *v26 == 289 )
        goto LABEL_11;
      v28 = v29;
    }
LABEL_12:
    v26 = (int *)(v24 + 40 * v23);
  }
LABEL_11:
  v15 = *((_QWORD *)v26 + 1);
  v14 = *((_QWORD *)v26 + 2);
LABEL_4:
  v16 = *(_QWORD *)(a4 + 8);
  v35 = v15;
  v36 = v14;
  v17 = sub_157E9C0(v16);
  v30 = sub_15A9620(a5, v17, 0);
  v31 = sub_16471D0(*(_QWORD **)(a4 + 24), 0);
  v32 = sub_16471D0(*(_QWORD **)(a4 + 24), 0);
  v18 = (__int64 *)sub_1643350(*(_QWORD **)(a4 + 24));
  v39[0] = v32;
  v39[1] = v31;
  v39[2] = v30;
  v38[1] = 0x300000003LL;
  v19 = sub_1644EA0(v18, v39, 3, 0);
  v20 = sub_1632080(v13, v35, v36, v19, 0);
  sub_1AB1740(v13, v35, v36, a6);
  LOWORD(v39[0]) = 261;
  v38[0] = &v35;
  v37[0] = sub_1AB1800(a1, (__int64 *)a4);
  v37[1] = sub_1AB1800(a2, (__int64 *)a4);
  v37[2] = a3;
  v6 = sub_1285290((__int64 *)a4, *(_QWORD *)(*(_QWORD *)v20 + 24LL), v20, (int)v37, 3, (__int64)v38, 0);
  v21 = sub_1649C60(v20);
  if ( !*(_BYTE *)(v21 + 16) )
    *(_WORD *)(v6 + 18) = *(_WORD *)(v6 + 18) & 0x8000 | *(_WORD *)(v6 + 18) & 3 | (*(_WORD *)(v21 + 18) >> 2) & 0xFFC;
  return v6;
}
