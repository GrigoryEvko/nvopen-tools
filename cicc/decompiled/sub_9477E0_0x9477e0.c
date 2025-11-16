// Function: sub_9477E0
// Address: 0x9477e0
//
__int64 __fastcall sub_9477E0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 *v7; // r12
  __int64 v8; // r15
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // rdi
  __int64 v14; // rax
  __int64 v15; // rdi
  __int64 v16; // r13
  __int64 v17; // rcx
  unsigned __int64 v18; // rdx
  unsigned __int64 v19; // r10
  unsigned int v20; // r11d
  char v21; // r13
  __int64 v22; // r15
  __int64 v23; // r9
  unsigned __int64 v24; // r8
  __int64 result; // rax
  __int64 v26; // rcx
  unsigned int v27; // r11d
  unsigned __int64 v28; // r10
  __int64 v29; // rdi
  __int64 v30; // rax
  unsigned __int64 v31; // rsi
  unsigned int v32; // eax
  __int64 v33; // [rsp-10h] [rbp-B0h]
  unsigned int v34; // [rsp+4h] [rbp-9Ch]
  unsigned __int64 v35; // [rsp+8h] [rbp-98h]
  unsigned int v36; // [rsp+10h] [rbp-90h]
  unsigned int v37; // [rsp+10h] [rbp-90h]
  unsigned __int64 v38; // [rsp+10h] [rbp-90h]
  unsigned __int64 v39; // [rsp+18h] [rbp-88h]
  unsigned __int64 v40; // [rsp+18h] [rbp-88h]
  unsigned int v41; // [rsp+18h] [rbp-88h]
  unsigned __int64 v42; // [rsp+18h] [rbp-88h]
  unsigned int v43; // [rsp+20h] [rbp-80h]
  unsigned __int64 v44; // [rsp+20h] [rbp-80h]
  unsigned int v45; // [rsp+20h] [rbp-80h]
  const char *v47; // [rsp+30h] [rbp-70h] BYREF
  unsigned __int64 v48; // [rsp+38h] [rbp-68h]
  unsigned int v49; // [rsp+48h] [rbp-58h]
  char v50; // [rsp+50h] [rbp-50h]
  char v51; // [rsp+51h] [rbp-4Fh]
  int v52; // [rsp+60h] [rbp-40h]

  v7 = *(unsigned __int64 **)(a2 + 72);
  v8 = v7[2];
  if ( !dword_4D04810
    || !(unsigned int)sub_731770((__int64)v7, 0, a3, a4, dword_4D04810, a6)
    && !(unsigned int)sub_731770(v8, 0, v9, v10, v11, v12) )
  {
    sub_926800((__int64)&v47, *a1, (__int64)v7);
    v21 = v52;
    v42 = v48;
    v37 = v49;
    v45 = (unsigned int)v47;
    result = sub_947E80(*a1, v8, v48, v49, v52 & 1);
    v26 = v45;
    v28 = v42;
    v27 = v37;
    if ( !v45 )
      goto LABEL_15;
LABEL_10:
    sub_91B8A0("unexpected aggregate source type!", (_DWORD *)(a2 + 36), 1);
  }
  v51 = 1;
  v13 = *a1;
  v47 = "agg.tmp";
  v50 = 3;
  v14 = sub_921D70(v13, *v7, (__int64)&v47, v10);
  v15 = *v7;
  v16 = *a1;
  v39 = v14;
  if ( *(char *)(*v7 + 142) >= 0 && *(_BYTE *)(v15 + 140) == 12 )
    v17 = (unsigned int)sub_8D4AB0(v15);
  else
    v17 = *(unsigned int *)(v15 + 136);
  sub_947E80(v16, v8, v39, v17, 0);
  sub_926800((__int64)&v47, *a1, (__int64)v7);
  v18 = *v7;
  v19 = v48;
  v20 = v49;
  v21 = v52;
  v43 = (unsigned int)v47;
  v22 = *a1;
  if ( *(char *)(*v7 + 142) >= 0 && *(_BYTE *)(v18 + 140) == 12 )
  {
    v34 = v49;
    v35 = v48;
    v38 = *v7;
    v32 = sub_8D4AB0(v18);
    v20 = v34;
    v19 = v35;
    v18 = v38;
    v23 = v32;
  }
  else
  {
    v23 = *(unsigned int *)(v18 + 136);
  }
  v24 = v39;
  v36 = v20;
  v40 = v19;
  result = sub_947440(v22, v19, v20, v21 & 1, v24, v23, 0, v18);
  v26 = v43;
  v27 = v36;
  v28 = v40;
  if ( v43 )
    goto LABEL_10;
LABEL_15:
  v31 = a1[2];
  if ( !v31 )
  {
    v41 = v27;
    v44 = v28;
    if ( (v21 & 1) == 0 )
      return result;
    v29 = *a1;
    v51 = 1;
    v47 = "agg.tmp";
    v50 = 3;
    v30 = sub_921D70(v29, *(_QWORD *)a2, (__int64)&v47, v26);
    v27 = v41;
    v28 = v44;
    a1[2] = v30;
    v31 = v30;
  }
  sub_947440(*a1, v31, *((_DWORD *)a1 + 6), *((unsigned __int8 *)a1 + 28), v28, v27, v21 & 1, *(_QWORD *)a2);
  return v33;
}
