// Function: sub_27262E0
// Address: 0x27262e0
//
__int64 __fastcall sub_27262E0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rax
  __int64 v4; // r12
  __int64 *v5; // rdx
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rdx
  unsigned __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 *v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  unsigned int v21; // ebx
  _QWORD v23[4]; // [rsp+10h] [rbp-720h] BYREF
  __int64 v24; // [rsp+30h] [rbp-700h]
  __int64 v25; // [rsp+38h] [rbp-6F8h]
  unsigned int v26; // [rsp+40h] [rbp-6F0h]
  __int64 *v27; // [rsp+48h] [rbp-6E8h]
  __int64 v28; // [rsp+50h] [rbp-6E0h]
  __int64 v29; // [rsp+58h] [rbp-6D8h] BYREF
  __int64 v30; // [rsp+60h] [rbp-6D0h]
  __int64 v31; // [rsp+68h] [rbp-6C8h]
  unsigned int v32; // [rsp+70h] [rbp-6C0h]
  _BYTE *v33; // [rsp+78h] [rbp-6B8h]
  __int64 v34; // [rsp+80h] [rbp-6B0h]
  _BYTE v35[1024]; // [rsp+88h] [rbp-6A8h] BYREF
  __int64 v36; // [rsp+488h] [rbp-2A8h]
  char *v37; // [rsp+490h] [rbp-2A0h]
  __int64 v38; // [rsp+498h] [rbp-298h]
  int v39; // [rsp+4A0h] [rbp-290h]
  char v40; // [rsp+4A4h] [rbp-28Ch]
  char v41; // [rsp+4A8h] [rbp-288h] BYREF
  __int64 v42; // [rsp+5A8h] [rbp-188h]
  __int64 v43; // [rsp+5B0h] [rbp-180h]
  __int64 v44; // [rsp+5B8h] [rbp-178h]
  __int64 v45; // [rsp+5C0h] [rbp-170h]
  _BYTE *v46; // [rsp+5C8h] [rbp-168h]
  __int64 v47; // [rsp+5D0h] [rbp-160h]
  _BYTE v48[128]; // [rsp+5D8h] [rbp-158h] BYREF
  __int64 v49; // [rsp+658h] [rbp-D8h]
  char *v50; // [rsp+660h] [rbp-D0h]
  __int64 v51; // [rsp+668h] [rbp-C8h]
  int v52; // [rsp+670h] [rbp-C0h]
  char v53; // [rsp+674h] [rbp-BCh]
  char v54; // [rsp+678h] [rbp-B8h] BYREF

  v2 = sub_B82360(*(_QWORD *)(a1 + 8), (__int64)&unk_4F8144C);
  if ( v2 && (v3 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v2 + 104LL))(v2, &unk_4F8144C)) != 0 )
    v4 = v3 + 176;
  else
    v4 = 0;
  v5 = *(__int64 **)(a1 + 8);
  v6 = *v5;
  v7 = v5[1];
  if ( v6 == v7 )
LABEL_20:
    BUG();
  while ( *(_UNKNOWN **)v6 != &unk_4F8FBD4 )
  {
    v6 += 16;
    if ( v7 == v6 )
      goto LABEL_20;
  }
  v8 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v6 + 8) + 104LL))(*(_QWORD *)(v6 + 8), &unk_4F8FBD4);
  v23[1] = v4;
  v23[0] = a2;
  v23[2] = v8 + 176;
  v34 = 0x8000000000LL;
  v37 = &v41;
  v47 = 0x1000000000LL;
  v50 = &v54;
  v23[3] = 0;
  v24 = 0;
  v25 = 0;
  v26 = 0;
  v27 = &v29;
  v28 = 0;
  v29 = 0;
  v30 = 0;
  v31 = 0;
  v32 = 0;
  v33 = v35;
  v36 = 0;
  v38 = 32;
  v39 = 0;
  v40 = 1;
  v42 = 0;
  v43 = 0;
  v44 = 0;
  v45 = 0;
  v46 = v48;
  v49 = 0;
  v51 = 16;
  v52 = 0;
  v53 = 1;
  sub_2723690((__int64)v23, (__int64)&unk_4F8FBD4, v9, v10, v11, v12);
  sub_2721090((__int64)v23, (__int64)&unk_4F8FBD4, v13, v14, v15, v16);
  v21 = sub_2722E50((__int64)v23, (__int64)&unk_4F8FBD4, v17, v18, v19, v20);
  if ( !v53 )
    _libc_free((unsigned __int64)v50);
  if ( v46 != v48 )
    _libc_free((unsigned __int64)v46);
  sub_C7D6A0(v43, 8LL * (unsigned int)v45, 8);
  if ( !v40 )
    _libc_free((unsigned __int64)v37);
  if ( v33 != v35 )
    _libc_free((unsigned __int64)v33);
  sub_C7D6A0(v30, 24LL * v32, 8);
  if ( v27 != &v29 )
    _libc_free((unsigned __int64)v27);
  sub_C7D6A0(v24, 16LL * v26, 8);
  return v21;
}
