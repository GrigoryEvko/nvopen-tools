// Function: sub_1A96660
// Address: 0x1a96660
//
void __fastcall sub_1A96660(__int64 *a1, __int64 *a2, __int64 a3, __int64 a4)
{
  _QWORD **v6; // r13
  __int64 *v7; // rax
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // r14
  __int64 v11; // r13
  __int64 v12; // r15
  _QWORD *v13; // rax
  int v14; // r8d
  int v15; // r9d
  __int64 v16; // r14
  int v17; // r8d
  __int64 v18; // r9
  _QWORD *v19; // rdx
  __int64 *v20; // rax
  __int64 v21; // rcx
  __int64 v22; // rax
  unsigned __int64 v23; // r14
  unsigned int v24; // r15d
  __int64 v25; // r9
  _QWORD *v26; // rax
  int v27; // r8d
  int v28; // r9d
  __int64 v29; // r11
  __int64 v30; // rdx
  __int64 v31; // r9
  _QWORD *v32; // rax
  _QWORD *v33; // rdx
  __int64 v34; // [rsp+0h] [rbp-70h]
  __int64 v35; // [rsp+8h] [rbp-68h]
  __int64 v36; // [rsp+8h] [rbp-68h]
  __int64 v37; // [rsp+10h] [rbp-60h]
  __int64 v38; // [rsp+10h] [rbp-60h]
  __int64 v39; // [rsp+10h] [rbp-60h]
  _QWORD *v40; // [rsp+10h] [rbp-60h]
  _BYTE v41[16]; // [rsp+20h] [rbp-50h] BYREF
  __int16 v42; // [rsp+30h] [rbp-40h]

  if ( !a3 )
    return;
  v6 = (_QWORD **)sub_15F2050(*a1 & 0xFFFFFFFFFFFFFFF8LL);
  v7 = (__int64 *)sub_1643270(*v6);
  v8 = sub_16453E0(v7, 1u);
  v9 = sub_1632190((__int64)v6, (__int64)"__tmp_use", 9, v8);
  v10 = *a1;
  v11 = v9;
  if ( (*a1 & 4) != 0 )
  {
    v12 = *(_QWORD *)((v10 & 0xFFFFFFFFFFFFFFF8LL) + 32);
    v42 = 257;
    if ( v12 )
      v12 -= 24;
    v37 = *(_QWORD *)(*(_QWORD *)v9 + 24LL);
    v13 = sub_1648AB0(72, (int)a3 + 1, 0);
    v16 = (__int64)v13;
    if ( !v13 )
      goto LABEL_8;
    v17 = a3 + 1;
    v18 = v12;
    v19 = &v13[-3 * a3];
    v20 = *(__int64 **)(v37 + 16);
    v21 = (__int64)(v19 - 3);
  }
  else
  {
    v23 = v10 & 0xFFFFFFFFFFFFFFF8LL;
    v24 = a3 + 1;
    v25 = sub_157EE30(*(_QWORD *)(v23 - 48));
    v42 = 257;
    if ( v25 )
      v25 -= 24;
    v34 = v25;
    v38 = *(_QWORD *)(*(_QWORD *)v11 + 24LL);
    v26 = sub_1648AB0(72, (int)a3 + 1, 0);
    if ( v26 )
    {
      v29 = v38;
      v39 = (__int64)v26;
      v35 = v29;
      sub_15F1EA0((__int64)v26, **(_QWORD **)(v29 + 16), 54, (__int64)&v26[-3 * a3 - 3], v24, v34);
      *(_QWORD *)(v39 + 56) = 0;
      sub_15F5B40(v39, v35, v11, a2, a3, (__int64)v41, 0, 0);
      v26 = (_QWORD *)v39;
    }
    v30 = *(unsigned int *)(a4 + 8);
    if ( (unsigned int)v30 >= *(_DWORD *)(a4 + 12) )
    {
      v40 = v26;
      sub_16CD150(a4, (const void *)(a4 + 16), 0, 8, v27, v28);
      v30 = *(unsigned int *)(a4 + 8);
      v26 = v40;
    }
    *(_QWORD *)(*(_QWORD *)a4 + 8 * v30) = v26;
    ++*(_DWORD *)(a4 + 8);
    v31 = sub_157EE30(*(_QWORD *)(v23 - 24));
    v42 = 257;
    if ( v31 )
      v31 -= 24;
    v36 = v31;
    v37 = *(_QWORD *)(*(_QWORD *)v11 + 24LL);
    v32 = sub_1648AB0(72, v24, 0);
    v16 = (__int64)v32;
    if ( !v32 )
      goto LABEL_8;
    v18 = v36;
    v17 = a3 + 1;
    v33 = &v32[-3 * a3];
    v20 = *(__int64 **)(v37 + 16);
    v21 = (__int64)(v33 - 3);
  }
  sub_15F1EA0(v16, *v20, 54, v21, v17, v18);
  *(_QWORD *)(v16 + 56) = 0;
  sub_15F5B40(v16, v37, v11, a2, a3, (__int64)v41, 0, 0);
LABEL_8:
  v22 = *(unsigned int *)(a4 + 8);
  if ( (unsigned int)v22 >= *(_DWORD *)(a4 + 12) )
  {
    sub_16CD150(a4, (const void *)(a4 + 16), 0, 8, v14, v15);
    v22 = *(unsigned int *)(a4 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a4 + 8 * v22) = v16;
  ++*(_DWORD *)(a4 + 8);
}
