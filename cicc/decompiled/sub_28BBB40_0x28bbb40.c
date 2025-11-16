// Function: sub_28BBB40
// Address: 0x28bbb40
//
__int64 __fastcall sub_28BBB40(__int64 a1, __int64 a2, int a3, __int64 a4)
{
  __int64 v4; // rax
  unsigned int v6; // eax
  bool v7; // cc
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rdx
  int v11; // eax
  unsigned int v12; // r15d
  unsigned int v13; // r9d
  __int64 v14; // rcx
  __int64 v15; // rdx
  unsigned int v16; // r13d
  __int64 v17; // r11
  __int64 v18; // r10
  unsigned int v19; // r14d
  unsigned __int64 v20; // r8
  int v21; // eax
  unsigned __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // [rsp+0h] [rbp-140h]
  __int64 v26; // [rsp+8h] [rbp-138h]
  unsigned __int64 v27; // [rsp+10h] [rbp-130h]
  unsigned int v28; // [rsp+1Ch] [rbp-124h]
  __int64 v29; // [rsp+20h] [rbp-120h]
  __int64 v30; // [rsp+28h] [rbp-118h]
  int v31; // [rsp+30h] [rbp-110h]
  unsigned __int64 v33; // [rsp+38h] [rbp-108h]
  _QWORD v34[2]; // [rsp+40h] [rbp-100h] BYREF
  __int64 v35; // [rsp+50h] [rbp-F0h] BYREF
  __int64 v36; // [rsp+58h] [rbp-E8h]
  unsigned int v37; // [rsp+60h] [rbp-E0h]
  unsigned __int64 v38; // [rsp+68h] [rbp-D8h]
  unsigned int v39; // [rsp+70h] [rbp-D0h]
  __int64 v40; // [rsp+80h] [rbp-C0h] BYREF
  __int64 v41; // [rsp+88h] [rbp-B8h]
  unsigned int v42; // [rsp+90h] [rbp-B0h]
  unsigned __int64 v43; // [rsp+98h] [rbp-A8h]
  unsigned int v44; // [rsp+A0h] [rbp-A0h]
  __int64 v45; // [rsp+B0h] [rbp-90h]
  __int64 v46; // [rsp+B8h] [rbp-88h]
  unsigned int v47; // [rsp+C0h] [rbp-80h]
  unsigned __int64 v48; // [rsp+C8h] [rbp-78h] BYREF
  unsigned int v49; // [rsp+D0h] [rbp-70h]
  __int64 v50; // [rsp+D8h] [rbp-68h]
  __int64 v51; // [rsp+E0h] [rbp-60h]
  unsigned int v52; // [rsp+E8h] [rbp-58h]
  unsigned __int64 v53; // [rsp+F0h] [rbp-50h] BYREF
  unsigned int v54; // [rsp+F8h] [rbp-48h]
  int v55; // [rsp+100h] [rbp-40h]
  __int64 v56; // [rsp+108h] [rbp-38h]

  v4 = *(_QWORD *)(a2 + 16);
  if ( !v4 || *(_QWORD *)(v4 + 8) || a3 != (*(_WORD *)(a2 + 2) & 0x3F) )
  {
    *(_BYTE *)(a1 + 96) = 0;
    return a1;
  }
  sub_28BB810((__int64)&v35, *(_QWORD *)(a2 - 64), a4);
  if ( v37 )
  {
    sub_28BB810((__int64)&v40, *(_QWORD *)(a2 - 32), a4);
    if ( !v42 )
    {
      v7 = v44 <= 0x40;
      *(_BYTE *)(a1 + 96) = 0;
      if ( !v7 && v43 )
        j_j___libc_free_0_0(v43);
      v6 = v39;
      goto LABEL_8;
    }
    v8 = sub_B43CC0(a2);
    v9 = sub_9208B0(v8, *(_QWORD *)(*(_QWORD *)(a2 - 64) + 8LL));
    v34[1] = v10;
    v34[0] = v9;
    v11 = sub_CA1930(v34);
    v12 = v44;
    v56 = a2;
    v13 = v39;
    v31 = v11;
    v14 = v40;
    v15 = v41;
    v44 = 0;
    v16 = v42;
    v17 = v35;
    v33 = v43;
    v18 = v36;
    v49 = v39;
    v19 = v37;
    v20 = v38;
    v50 = v40;
    v39 = 0;
    v45 = v35;
    v46 = v36;
    v47 = v37;
    v48 = v38;
    v51 = v41;
    v52 = v42;
    v54 = v12;
    v53 = v43;
    v55 = v11;
    if ( v37 == v42 )
    {
      v25 = v41;
      v26 = v40;
      v27 = v38;
      v28 = v13;
      v29 = v36;
      v30 = v35;
      v21 = sub_C4C880((__int64)&v53, (__int64)&v48);
      v17 = v30;
      v18 = v29;
      v13 = v28;
      v20 = v27;
      v14 = v26;
      v15 = v25;
      if ( v21 >= 0 )
        goto LABEL_18;
    }
    else if ( v37 <= v42 )
    {
LABEL_18:
      *(_QWORD *)a1 = v17;
      *(_QWORD *)(a1 + 8) = v18;
      *(_QWORD *)(a1 + 64) = v33;
      *(_DWORD *)(a1 + 16) = v19;
      *(_DWORD *)(a1 + 32) = v13;
      *(_QWORD *)(a1 + 24) = v20;
      *(_QWORD *)(a1 + 40) = v14;
      *(_QWORD *)(a1 + 48) = v15;
      *(_DWORD *)(a1 + 56) = v16;
      *(_DWORD *)(a1 + 72) = v12;
      *(_DWORD *)(a1 + 80) = v31;
      *(_QWORD *)(a1 + 88) = a2;
      *(_BYTE *)(a1 + 96) = 1;
      return a1;
    }
    v22 = v20;
    v20 = v33;
    v33 = v22;
    LODWORD(v22) = v13;
    v13 = v12;
    v12 = v22;
    LODWORD(v22) = v19;
    v19 = v16;
    v16 = v22;
    v23 = v18;
    v18 = v15;
    v15 = v23;
    v24 = v17;
    v17 = v14;
    v14 = v24;
    goto LABEL_18;
  }
  *(_BYTE *)(a1 + 96) = 0;
  v6 = v39;
LABEL_8:
  if ( v6 > 0x40 && v38 )
    j_j___libc_free_0_0(v38);
  return a1;
}
