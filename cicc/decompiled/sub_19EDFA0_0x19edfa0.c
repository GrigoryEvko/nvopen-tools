// Function: sub_19EDFA0
// Address: 0x19edfa0
//
__int64 __fastcall sub_19EDFA0(__int64 a1, __int64 a2)
{
  int v3; // ebx
  __int64 v4; // rax
  int v5; // ebx
  __int64 v6; // r12
  __int64 *v7; // r8
  char v8; // r15
  int v9; // r9d
  int v10; // ebx
  unsigned __int64 *v11; // r8
  unsigned __int64 v12; // rbx
  unsigned int v13; // r15d
  bool v14; // al
  __int64 v15; // rax
  int v16; // r8d
  int v17; // r9d
  __int64 v18; // rax
  unsigned __int64 *v20; // r8
  bool v21; // al
  unsigned int v22; // eax
  __int64 *v23; // rax
  unsigned __int64 v24; // rdx
  unsigned __int64 v25; // rsi
  __int64 *v26; // rbx
  __int64 v27; // rax
  __int64 *v28; // r9
  __int64 v29; // r8
  __int64 v30; // rdx
  __int64 *v31; // rbx
  _BYTE *v32; // rax
  _QWORD *v33; // rax
  __int64 *v34; // rsi
  __int64 v35; // rcx
  int v36; // r8d
  int v37; // r9d
  __int64 v38; // rbx
  unsigned __int64 v39; // [rsp+0h] [rbp-A0h]
  unsigned __int8 v40; // [rsp+8h] [rbp-98h]
  unsigned __int64 v41; // [rsp+8h] [rbp-98h]
  __int64 *v42; // [rsp+8h] [rbp-98h]
  __int64 *v43; // [rsp+10h] [rbp-90h]
  char v44; // [rsp+10h] [rbp-90h]
  __int64 v45; // [rsp+10h] [rbp-90h]
  unsigned __int64 v46; // [rsp+18h] [rbp-88h]
  __int64 *v47; // [rsp+18h] [rbp-88h]
  _BYTE *v48; // [rsp+20h] [rbp-80h] BYREF
  __int64 v49; // [rsp+28h] [rbp-78h]
  _BYTE v50[112]; // [rsp+30h] [rbp-70h] BYREF

  v3 = *(_DWORD *)(a2 + 20);
  v4 = sub_145CDC0(0x30u, (__int64 *)(a1 + 64));
  v5 = v3 & 0xFFFFFFF;
  v6 = v4;
  if ( v4 )
  {
    *(_DWORD *)(v4 + 32) = v5;
    *(_QWORD *)(v4 + 16) = 0;
    *(_QWORD *)(v4 + 24) = 0;
    *(_QWORD *)v4 = &unk_49F4D90;
    *(_QWORD *)(v4 + 8) = 0xFFFFFFFD00000006LL;
    *(_DWORD *)(v4 + 36) = 0;
    *(_QWORD *)(v4 + 40) = 0;
  }
  v8 = sub_19E5840(a1, a2, v4);
  v9 = *(unsigned __int8 *)(a2 + 16);
  v10 = v9 - 24;
  if ( (unsigned int)(v9 - 24) <= 0x1C && ((1LL << v10) & 0x1C019800) != 0 )
  {
    v20 = *(unsigned __int64 **)(v6 + 24);
    v44 = *(_BYTE *)(a2 + 16);
    v47 = (__int64 *)v20;
    v39 = v20[1];
    v41 = *v20;
    v21 = sub_19E5280(a1, *v20, v39);
    v7 = v47;
    LOBYTE(v9) = v44;
    if ( v21 )
    {
      *v47 = v39;
      v47[1] = v41;
      LOBYTE(v9) = *(_BYTE *)(a2 + 16);
      if ( (unsigned __int8)(v9 - 75) <= 1u )
      {
LABEL_6:
        v11 = *(unsigned __int64 **)(v6 + 24);
        v40 = v9;
        v12 = *v11;
        v43 = (__int64 *)v11;
        v13 = *(_WORD *)(a2 + 18) & 0x7FFF;
        v46 = v11[1];
        v14 = sub_19E5280(a1, *v11, v46);
        v7 = v43;
        if ( v14 )
        {
          v43[1] = v12;
          *v43 = v46;
          v22 = sub_15FF5D0(v13);
          v7 = *(__int64 **)(v6 + 24);
          v13 = v22;
          v10 = *(unsigned __int8 *)(a2 + 16) - 24;
        }
        else
        {
          v10 = v40 - 24;
        }
        goto LABEL_8;
      }
    }
    else if ( (unsigned __int8)(v44 - 75) <= 1u )
    {
      v13 = *(_WORD *)(a2 + 18) & 0x7FFF;
LABEL_8:
      *(_DWORD *)(v6 + 12) = v13 | (v10 << 8);
      v15 = (__int64)sub_13E1240(v13, *v7, v7[1], (__int64 *)(a1 + 1352));
      goto LABEL_9;
    }
  }
  else if ( (unsigned __int8)(v9 - 75) <= 1u )
  {
    goto LABEL_6;
  }
  if ( (_BYTE)v9 == 79 )
  {
    v23 = *(__int64 **)(v6 + 24);
    v24 = v23[2];
    v25 = v23[1];
    if ( *(_BYTE *)(*v23 + 16) > 0x10u && v24 != v25 )
      return v6;
    v15 = sub_13E2B90(*v23, v25, v24, (__int64 *)(a1 + 1352));
    goto LABEL_9;
  }
  if ( (unsigned int)(unsigned __int8)v9 - 35 <= 0x11 )
  {
    v15 = (__int64)sub_13E1140(
                     *(_DWORD *)(v6 + 12),
                     **(unsigned __int8 ***)(v6 + 24),
                     *(unsigned __int8 **)(*(_QWORD *)(v6 + 24) + 8LL),
                     (_QWORD *)(a1 + 1352));
    goto LABEL_9;
  }
  if ( (_BYTE)v9 == 71 )
  {
    v15 = sub_13D1870(47, *(__int64 **)(a2 - 24), *(_QWORD *)a2, (_QWORD *)(a1 + 1352), (__int64)v7);
    goto LABEL_9;
  }
  if ( (_BYTE)v9 == 56 )
  {
    v15 = sub_13E3340(*(_QWORD *)(v6 + 40), *(__int64 ***)(v6 + 24), *(unsigned int *)(v6 + 36), (__int64 *)(a1 + 1352));
LABEL_9:
    v18 = sub_19EDC00(a1, v6, a2, v15, v16, v17);
    if ( v18 )
      return v18;
    return v6;
  }
  if ( v8 )
  {
    v26 = *(__int64 **)(v6 + 24);
    v49 = 0x800000000LL;
    v27 = *(unsigned int *)(v6 + 36);
    v48 = v50;
    v28 = &v26[v27];
    if ( v26 == v28 )
    {
      v30 = 0;
      v34 = (__int64 *)v50;
    }
    else
    {
      v29 = *v26;
      LODWORD(v30) = 0;
      v31 = v26 + 1;
      v32 = v50;
      while ( 1 )
      {
        v33 = &v32[8 * (unsigned int)v30];
        if ( v33 )
        {
          *v33 = v29;
          LODWORD(v30) = v49;
        }
        v30 = (unsigned int)(v30 + 1);
        LODWORD(v49) = v30;
        if ( v28 == v31 )
          break;
        v29 = *v31;
        if ( HIDWORD(v49) <= (unsigned int)v30 )
        {
          v42 = v28;
          v45 = *v31;
          sub_16CD150((__int64)&v48, v50, 0, 8, v29, (int)v28);
          LODWORD(v30) = v49;
          v28 = v42;
          v29 = v45;
        }
        v32 = v48;
        ++v31;
      }
      v34 = (__int64 *)v48;
    }
    v35 = sub_14DD1F0(a2, v34, v30, *(_BYTE **)(a1 + 48), *(_QWORD *)(a1 + 16));
    if ( v35 && (v38 = sub_19EDC00(a1, v6, a2, v35, v36, v37)) != 0 )
    {
      if ( v48 != v50 )
        _libc_free((unsigned __int64)v48);
      return v38;
    }
    else if ( v48 != v50 )
    {
      _libc_free((unsigned __int64)v48);
    }
  }
  return v6;
}
