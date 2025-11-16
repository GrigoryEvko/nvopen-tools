// Function: sub_30302D0
// Address: 0x30302d0
//
__int64 __fastcall sub_30302D0(__int128 a1, __int64 a2, int a3, int a4, int a5, __int64 a6)
{
  bool v6; // zf
  __int64 *v8; // rax
  __int64 v12; // rcx
  __int64 v13; // r14
  __int64 v14; // r15
  int v15; // edx
  __int64 v16; // rcx
  int v17; // edx
  __int64 v18; // rcx
  _QWORD *v19; // rdx
  __int64 v20; // r8
  __int64 v21; // r9
  int v22; // r9d
  __int64 v23; // rdx
  __int128 *v24; // rdx
  __int64 v25; // rdi
  __int128 *v26; // rax
  __int64 v27; // rcx
  _QWORD *v28; // rdx
  __int64 v29; // r8
  __int64 v30; // r9
  __int64 v31; // rdx
  __int128 v32; // [rsp-30h] [rbp-80h]
  __int128 v33; // [rsp-10h] [rbp-60h]
  __int128 v34; // [rsp-10h] [rbp-60h]
  __int128 v35; // [rsp+0h] [rbp-50h] BYREF
  __int64 v36; // [rsp+10h] [rbp-40h] BYREF
  __int64 v37; // [rsp+18h] [rbp-38h]

  v6 = *(_DWORD *)(a2 + 24) == 205;
  v35 = a1;
  if ( !v6 )
    return 0;
  v8 = *(__int64 **)(a2 + 40);
  v12 = v8[5];
  v13 = *v8;
  v14 = v8[1];
  v15 = *(_DWORD *)(v12 + 24);
  if ( v15 != 35 && v15 != 11 )
    goto LABEL_5;
  v27 = *(_QWORD *)(v12 + 96);
  v28 = *(_QWORD **)(v27 + 24);
  if ( *(_DWORD *)(v27 + 32) > 0x40u )
    v28 = (_QWORD *)*v28;
  if ( v28 == (_QWORD *)1 )
  {
    if ( !sub_30301B0(v8[10]) )
      return 0;
    *((_QWORD *)&v34 + 1) = v30;
    *(_QWORD *)&v34 = v29;
    v36 = sub_3406EB0(*(_QWORD *)(a6 + 16), 58, a5, a3, a4, v30, v35, v34);
    v37 = v31;
    v24 = (__int128 *)&v36;
    v25 = *(_QWORD *)(a6 + 16);
    v26 = &v35;
  }
  else
  {
LABEL_5:
    v16 = v8[10];
    v17 = *(_DWORD *)(v16 + 24);
    if ( v17 != 35 && v17 != 11 )
      return 0;
    v18 = *(_QWORD *)(v16 + 96);
    v19 = *(_QWORD **)(v18 + 24);
    if ( *(_DWORD *)(v18 + 32) > 0x40u )
      v19 = (_QWORD *)*v19;
    if ( v19 != (_QWORD *)1 || !sub_30301B0(v8[5]) )
      return 0;
    *((_QWORD *)&v33 + 1) = v21;
    *(_QWORD *)&v33 = v20;
    v36 = sub_3406EB0(*(_QWORD *)(a6 + 16), 58, a5, a3, a4, v21, v35, v33);
    v37 = v23;
    v24 = &v35;
    v25 = *(_QWORD *)(a6 + 16);
    v26 = (__int128 *)&v36;
  }
  *((_QWORD *)&v32 + 1) = v14;
  *(_QWORD *)&v32 = v13;
  return sub_340F900(v25, 205, a5, a3, a4, v22, v32, *v26, *v24);
}
