// Function: sub_3030880
// Address: 0x3030880
//
__int64 __fastcall sub_3030880(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v6; // rax
  __int64 v7; // r15
  int v8; // edx
  __int64 v9; // rcx
  __int64 v10; // rsi
  _QWORD *v11; // rdx
  int v12; // r8d
  __int64 *v13; // rax
  __int64 v14; // r13
  __int64 v15; // rsi
  int v16; // eax
  __int64 v17; // rdx
  __int64 v18; // rdx
  _QWORD *v19; // rax
  __int64 v20; // rcx
  __int64 v21; // rbx
  __int64 v22; // rsi
  __int64 v24; // rax
  _QWORD *v25; // rdi
  int v26; // eax
  __int64 v27; // r15
  int v28; // edx
  int v29; // ebx
  unsigned __int16 *v30; // r13
  __int64 v31; // r8
  __int64 v32; // rcx
  __int64 v33; // r14
  int v34; // edx
  int v35; // r13d
  __int64 v36; // rax
  int v37; // edx
  int v38; // eax
  int v39; // [rsp+Ch] [rbp-74h]
  int v40; // [rsp+Ch] [rbp-74h]
  __int64 v41; // [rsp+10h] [rbp-70h]
  __int64 v42; // [rsp+10h] [rbp-70h]
  __int64 v43; // [rsp+18h] [rbp-68h]
  __int64 v44; // [rsp+28h] [rbp-58h] BYREF
  __int64 v45; // [rsp+30h] [rbp-50h] BYREF
  int v46; // [rsp+38h] [rbp-48h]
  __int64 v47; // [rsp+40h] [rbp-40h] BYREF
  int v48; // [rsp+48h] [rbp-38h]

  v6 = *(_QWORD *)(a2 + 40);
  v7 = *(_QWORD *)(v6 + 80);
  v8 = *(_DWORD *)(v7 + 24);
  if ( v8 != 35 && v8 != 11 )
    sub_C64ED0("The first argument of load/store intrinsic must be a constant.", 1u);
  v9 = *(_QWORD *)(v6 + 40);
  v10 = *(_QWORD *)(v9 + 96);
  v11 = *(_QWORD **)(v10 + 24);
  if ( *(_DWORD *)(v10 + 32) > 0x40u )
    v11 = (_QWORD *)*v11;
  v12 = *(_DWORD *)(v6 + 88);
  v13 = (__int64 *)(v6 + 40LL * (unsigned int)(*(_DWORD *)(a2 + 64) - 1));
  v14 = *((unsigned int *)v13 + 2);
  v43 = *v13;
  if ( v11 == (_QWORD *)8938 || v11 == (_QWORD *)9553 )
  {
    *(_QWORD *)a1 = v7;
    *(_DWORD *)(a1 + 8) = v12;
    *(_QWORD *)(a1 + 16) = v43;
    *(_DWORD *)(a1 + 24) = v14;
    return a1;
  }
  v15 = *(_QWORD *)(a2 + 80);
  v45 = v15;
  if ( v15 )
  {
    v39 = v12;
    v41 = v9;
    sub_B96E90((__int64)&v45, v15, 1);
    v12 = v39;
    v9 = v41;
  }
  v16 = *(_DWORD *)(a2 + 72);
  v17 = *(_QWORD *)(*(_QWORD *)(a3 + 40) + 16LL);
  v46 = v16;
  if ( *(_DWORD *)(v17 + 344) <= 0x4Fu )
  {
    v24 = *(_QWORD *)(v7 + 96);
    v25 = *(_QWORD **)(v24 + 24);
    if ( *(_DWORD *)(v24 + 32) > 0x40u )
      v25 = (_QWORD *)*v25;
    v44 = sub_CE1160((__int64)v25);
    if ( (v44 & 0x1000000000LL) != 0 )
    {
      v44 = sub_CE1160(0);
      BYTE4(v44) |= 0x10u;
    }
    BYTE1(v44) &= ~4u;
    v26 = sub_CE1170((__int64)&v44);
    v27 = sub_3400BD0(a3, v26, (unsigned int)&v45, 8, 0, 1, 0);
    v29 = v28;
    v30 = (unsigned __int16 *)(*(_QWORD *)(v43 + 48) + 16 * v14);
    v31 = *((_QWORD *)v30 + 1);
    v32 = *v30;
    v48 = 0;
    v47 = 0;
    v33 = sub_33F17F0(a3, 51, &v47, v32, v31);
    v35 = v34;
    if ( v47 )
      sub_B91220((__int64)&v47, v47);
    *(_QWORD *)a1 = v27;
    *(_DWORD *)(a1 + 8) = v29;
    *(_QWORD *)(a1 + 16) = v33;
    *(_DWORD *)(a1 + 24) = v35;
    goto LABEL_30;
  }
  v47 = v45;
  if ( v45 )
  {
    v40 = v12;
    v42 = v9;
    sub_B96E90((__int64)&v47, v45, 1);
    v16 = v46;
    v12 = v40;
    v9 = v42;
  }
  v48 = v16;
  v18 = *(_QWORD *)(v9 + 96);
  v19 = *(_QWORD **)(v18 + 24);
  if ( *(_DWORD *)(v18 + 32) > 0x40u )
    v19 = (_QWORD *)*v19;
  v20 = *(_QWORD *)(v7 + 96);
  v21 = *(_QWORD *)(v20 + 24);
  if ( *(_DWORD *)(v20 + 32) > 0x40u )
    v21 = **(_QWORD **)(v20 + 24);
  if ( (_DWORD)v19 == 8937 )
  {
    if ( (v21 & 0x1000000000LL) == 0 )
      goto LABEL_19;
    v36 = sub_3400BD0(a3, 0, (unsigned int)&v47, 8, 0, 1, 0);
  }
  else
  {
    if ( ((unsigned int)v19 & 0xFFFFFFFB) != 0x22D8 )
      goto LABEL_19;
    v44 = sub_CE1160(0);
    BYTE1(v44) = BYTE1(v44) & 0xFA | v21 & 1 | 4;
    v38 = sub_CE1170((__int64)&v44);
    v36 = sub_3400BD0(a3, v38, (unsigned int)&v47, 8, 0, 1, 0);
  }
  v7 = v36;
  v12 = v37;
LABEL_19:
  v22 = v47;
  *(_QWORD *)a1 = v7;
  *(_DWORD *)(a1 + 8) = v12;
  *(_QWORD *)(a1 + 16) = v43;
  *(_DWORD *)(a1 + 24) = v14;
  if ( v22 )
    sub_B91220((__int64)&v47, v22);
LABEL_30:
  if ( v45 )
    sub_B91220((__int64)&v45, v45);
  return a1;
}
