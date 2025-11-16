// Function: sub_DD9AA0
// Address: 0xdd9aa0
//
__int64 *__fastcall sub_DD9AA0(__int64 *a1, __int64 a2)
{
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rcx
  _QWORD *v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rdi
  unsigned int v11; // edx
  __int64 v12; // r14
  unsigned int v13; // esi
  __int64 v14; // r8
  _QWORD *v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rdi
  unsigned int v18; // eax
  __int64 v19; // rax
  __int64 v20; // rdx
  unsigned int v21; // eax
  __int64 v22; // rdx
  unsigned __int64 v23; // rax
  int v24; // ecx
  __int64 v25; // rcx
  __int64 v26; // rbx
  __int64 v27; // rax
  __int64 v28; // rbx
  char v29; // al
  __int64 v30; // r9
  char v31; // al
  _BYTE *v32; // r14
  _BYTE *v33; // rbx
  __int64 *v34; // rax
  __int64 *v35; // rax
  __int64 v36; // [rsp-68h] [rbp-68h]
  __int64 v37; // [rsp-68h] [rbp-68h]
  __int64 v38; // [rsp-60h] [rbp-60h]
  __int64 v39[2]; // [rsp-58h] [rbp-58h] BYREF
  __int64 v40[9]; // [rsp-48h] [rbp-48h] BYREF

  if ( (*(_DWORD *)(a2 + 4) & 0x7FFFFFF) != 2 )
    return 0;
  v5 = *(_QWORD *)(a2 - 8);
  v6 = 32LL * *(unsigned int *)(a2 + 72);
  v7 = v5 + v6 + 16;
  v8 = (_QWORD *)(v6 + v5);
  v9 = *v8;
  if ( *v8 )
  {
    v10 = (unsigned int)(*(_DWORD *)(v9 + 44) + 1);
    v11 = *(_DWORD *)(v9 + 44) + 1;
  }
  else
  {
    v10 = 0;
    v11 = 0;
  }
  v12 = a1[5];
  v13 = *(_DWORD *)(v12 + 32);
  if ( v11 < v13 && (v14 = *(_QWORD *)(v12 + 24), *(_QWORD *)(v14 + 8 * v10)) )
  {
    v15 = v8 + 1;
    v16 = v8[1];
    if ( v16 )
    {
      v17 = (unsigned int)(*(_DWORD *)(v16 + 44) + 1);
      v18 = *(_DWORD *)(v16 + 44) + 1;
    }
    else
    {
      v17 = 0;
      v18 = 0;
    }
    if ( v13 > v18 && *(_QWORD *)(v14 + 8 * v17) )
      goto LABEL_12;
  }
  else
  {
    v15 = v8;
  }
  if ( (_QWORD *)v7 != v15 )
    return 0;
LABEL_12:
  v19 = *(_QWORD *)(a2 + 40);
  if ( v19 )
  {
    v20 = (unsigned int)(*(_DWORD *)(v19 + 44) + 1);
    v21 = *(_DWORD *)(v19 + 44) + 1;
  }
  else
  {
    v20 = 0;
    v21 = 0;
  }
  if ( v13 <= v21 )
    BUG();
  v22 = **(_QWORD **)(*(_QWORD *)(*(_QWORD *)(v12 + 24) + 8 * v20) + 8LL);
  v23 = *(_QWORD *)(v22 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v23 == v22 + 48 )
    goto LABEL_36;
  if ( !v23 )
    BUG();
  v24 = *(unsigned __int8 *)(v23 - 24);
  if ( (unsigned int)(v24 - 30) > 0xA )
LABEL_36:
    BUG();
  if ( (_BYTE)v24 != 31 )
    return 0;
  if ( (*(_DWORD *)(v23 - 20) & 0x7FFFFFF) != 3 )
    return 0;
  v25 = *(_QWORD *)(v23 - 56);
  v26 = *(_QWORD *)(v23 - 120);
  v39[0] = *(_QWORD *)(v23 + 16);
  v39[1] = v25;
  v27 = *(_QWORD *)(v23 - 88);
  v38 = v26;
  v40[0] = v39[0];
  v40[1] = v27;
  if ( !(unsigned __int8)sub_B190C0(v39) )
    return 0;
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v28 = *(_QWORD *)(a2 - 8);
  else
    v28 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  v29 = sub_B19ED0(v12, v39, v28);
  v30 = v28 + 32;
  if ( v29 && (v31 = sub_B19ED0(v12, v40, v28 + 32), v30 = v28 + 32, v31) )
  {
    v32 = *(_BYTE **)v28;
    v33 = *(_BYTE **)(v28 + 32);
  }
  else
  {
    if ( !(unsigned __int8)sub_B19ED0(v12, v39, v30) || !(unsigned __int8)sub_B19ED0(v12, v40, v28) )
      return 0;
    v32 = *(_BYTE **)(v28 + 32);
    v33 = *(_BYTE **)v28;
  }
  v36 = *(_QWORD *)(a2 + 40);
  v34 = sub_DD8400((__int64)a1, (__int64)v32);
  if ( !sub_DAEB50((__int64)a1, (__int64)v34, v36) )
    return 0;
  v37 = *(_QWORD *)(a2 + 40);
  v35 = sub_DD8400((__int64)a1, (__int64)v33);
  if ( !sub_DAEB50((__int64)a1, (__int64)v35, v37) )
    return 0;
  return sub_DD99C0(a1, a2, v38, v32, v33);
}
