// Function: sub_18E3EF0
// Address: 0x18e3ef0
//
__int64 __fastcall sub_18E3EF0(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 *v6; // rdx
  __int64 v7; // r12
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  unsigned int v19; // eax
  unsigned int v20; // r12d
  _QWORD v22[4]; // [rsp+0h] [rbp-720h] BYREF
  __int64 v23; // [rsp+20h] [rbp-700h]
  __int64 v24; // [rsp+28h] [rbp-6F8h]
  int v25; // [rsp+30h] [rbp-6F0h]
  __int64 v26; // [rsp+38h] [rbp-6E8h]
  __int64 v27; // [rsp+40h] [rbp-6E0h]
  __int64 v28; // [rsp+48h] [rbp-6D8h]
  __int64 v29; // [rsp+50h] [rbp-6D0h]
  __int64 v30; // [rsp+58h] [rbp-6C8h]
  __int64 v31; // [rsp+60h] [rbp-6C0h]
  int v32; // [rsp+68h] [rbp-6B8h]
  _BYTE *v33; // [rsp+70h] [rbp-6B0h]
  __int64 v34; // [rsp+78h] [rbp-6A8h]
  _BYTE v35[1024]; // [rsp+80h] [rbp-6A0h] BYREF
  __int64 v36; // [rsp+480h] [rbp-2A0h]
  _BYTE *v37; // [rsp+488h] [rbp-298h]
  _BYTE *v38; // [rsp+490h] [rbp-290h]
  __int64 v39; // [rsp+498h] [rbp-288h]
  int v40; // [rsp+4A0h] [rbp-280h]
  _BYTE v41[256]; // [rsp+4A8h] [rbp-278h] BYREF
  __int64 v42; // [rsp+5A8h] [rbp-178h]
  _BYTE *v43; // [rsp+5B0h] [rbp-170h]
  _BYTE *v44; // [rsp+5B8h] [rbp-168h]
  __int64 v45; // [rsp+5C0h] [rbp-160h]
  int v46; // [rsp+5C8h] [rbp-158h]
  _BYTE v47[128]; // [rsp+5D0h] [rbp-150h] BYREF
  __int64 v48; // [rsp+650h] [rbp-D0h]
  _BYTE *v49; // [rsp+658h] [rbp-C8h]
  _BYTE *v50; // [rsp+660h] [rbp-C0h]
  __int64 v51; // [rsp+668h] [rbp-B8h]
  int v52; // [rsp+670h] [rbp-B0h]
  _BYTE v53[168]; // [rsp+678h] [rbp-A8h] BYREF

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_24:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_4F9E06C )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_24;
  }
  v5 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_4F9E06C);
  v6 = *(__int64 **)(a1 + 8);
  v7 = v5 + 160;
  v8 = *v6;
  v9 = v6[1];
  if ( v8 == v9 )
LABEL_23:
    BUG();
  while ( *(_UNKNOWN **)v8 != &unk_4F99CCC )
  {
    v8 += 16;
    if ( v9 == v8 )
      goto LABEL_23;
  }
  v10 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v8 + 8) + 104LL))(*(_QWORD *)(v8 + 8), &unk_4F99CCC);
  v22[1] = v7;
  v22[0] = a2;
  v22[2] = v10 + 160;
  v34 = 0x8000000000LL;
  v37 = v41;
  v38 = v41;
  v43 = v47;
  v44 = v47;
  v49 = v53;
  v50 = v53;
  v22[3] = 0;
  v23 = 0;
  v24 = 0;
  v25 = 0;
  v26 = 0;
  v27 = 0;
  v28 = 0;
  v29 = 0;
  v30 = 0;
  v31 = 0;
  v32 = 0;
  v33 = v35;
  v36 = 0;
  v39 = 32;
  v40 = 0;
  v42 = 0;
  v45 = 16;
  v46 = 0;
  v48 = 0;
  v51 = 16;
  v52 = 0;
  sub_18DFED0((__int64)v22, (__int64)&unk_4F99CCC, v11, v12, v13, v14);
  sub_18DFCA0((__int64)v22, (__int64)&unk_4F99CCC, v15, v16, v17, v18);
  LOBYTE(v19) = sub_18E25B0((__int64)v22);
  v20 = v19;
  if ( v50 != v49 )
    _libc_free((unsigned __int64)v50);
  if ( v44 != v43 )
    _libc_free((unsigned __int64)v44);
  if ( v38 != v37 )
    _libc_free((unsigned __int64)v38);
  if ( v33 != v35 )
    _libc_free((unsigned __int64)v33);
  j___libc_free_0(v30);
  if ( v26 )
    j_j___libc_free_0(v26, v28 - v26);
  j___libc_free_0(v23);
  return v20;
}
