// Function: sub_24D3240
// Address: 0x24d3240
//
unsigned __int64 *__fastcall sub_24D3240(unsigned int ***a1)
{
  __int64 *v2; // r12
  __int64 v3; // r14
  __int64 v4; // rax
  char v5; // al
  _QWORD *v6; // rax
  __int64 v7; // r9
  __int64 v8; // rbx
  __int64 v9; // rax
  __int64 v10; // r12
  __int64 v11; // r15
  __int64 v12; // r12
  __int64 v13; // rdx
  unsigned int v14; // esi
  unsigned __int64 *result; // rax
  unsigned __int64 v16; // rsi
  unsigned int **v17; // r12
  __int64 ***v18; // rax
  __int64 **v19; // r15
  unsigned int **v20; // rax
  unsigned int *v21; // rcx
  __int64 v22; // rdi
  _BYTE *v23; // rax
  unsigned __int64 v24; // rax
  unsigned __int64 v25; // rax
  unsigned int **v26; // r12
  __int64 **v27; // r15
  unsigned __int64 v28; // rax
  unsigned __int64 v29; // rax
  unsigned int **v30; // r12
  __int64 v31; // r15
  __int64 v32; // rax
  char v33; // al
  __int16 v34; // cx
  _QWORD *v35; // rax
  __int64 v36; // r9
  __int64 v37; // rbx
  unsigned int *v38; // rax
  __int64 v39; // r12
  unsigned int *v40; // r15
  __int64 v41; // rdx
  unsigned int v42; // esi
  __int64 v43; // [rsp+10h] [rbp-120h]
  char v44; // [rsp+18h] [rbp-118h]
  __int64 v45; // [rsp+20h] [rbp-110h]
  __int16 v46; // [rsp+2Eh] [rbp-102h]
  __int64 v47; // [rsp+30h] [rbp-100h] BYREF
  int v48; // [rsp+38h] [rbp-F8h]
  _QWORD v49[4]; // [rsp+40h] [rbp-F0h] BYREF
  __int16 v50; // [rsp+60h] [rbp-D0h]
  _QWORD v51[4]; // [rsp+70h] [rbp-C0h] BYREF
  __int16 v52; // [rsp+90h] [rbp-A0h]
  _QWORD v53[4]; // [rsp+A0h] [rbp-90h] BYREF
  __int16 v54; // [rsp+C0h] [rbp-70h]
  _QWORD v55[2]; // [rsp+D0h] [rbp-60h] BYREF
  __int64 *v56; // [rsp+E0h] [rbp-50h]
  __int16 v57; // [rsp+F0h] [rbp-40h]

  v2 = (__int64 *)*a1;
  v3 = (__int64)*a1[1];
  v43 = (__int64)*a1[2];
  v4 = sub_AA4E30((__int64)(*a1)[6]);
  v5 = sub_AE5020(v4, *(_QWORD *)(v3 + 8));
  v57 = 257;
  v44 = v5;
  v6 = sub_BD2C40(80, unk_3F10A10);
  v8 = (__int64)v6;
  if ( v6 )
    sub_B4D3C0((__int64)v6, v3, v43, 0, v44, v7, 0, 0);
  (*(void (__fastcall **)(__int64, __int64, _QWORD *, __int64, __int64))(*(_QWORD *)v2[11] + 16LL))(
    v2[11],
    v8,
    v55,
    v2[7],
    v2[8]);
  v9 = *v2;
  v10 = 16LL * *((unsigned int *)v2 + 2);
  v11 = v9 + v10;
  if ( v9 != v9 + v10 )
  {
    v12 = v9;
    do
    {
      v13 = *(_QWORD *)(v12 + 8);
      v14 = *(_DWORD *)v12;
      v12 += 16;
      sub_B99FD0(v8, v14, v13);
    }
    while ( v11 != v12 );
  }
  v47 = 1;
  result = (unsigned __int64 *)a1[3];
  if ( *result > 1 )
  {
    v16 = 1;
    do
    {
      v54 = 2819;
      v55[0] = v53;
      v17 = *a1;
      v56 = (__int64 *)".ptr";
      v18 = (__int64 ***)a1[6];
      v53[2] = &v47;
      v57 = 770;
      v53[0] = "shadow.byte.";
      v19 = *v18;
      v49[0] = "shadow.byte.";
      v52 = 770;
      v51[0] = v49;
      v51[2] = ".offset";
      v20 = a1[5];
      v49[2] = &v47;
      v21 = v20[10];
      v22 = (__int64)v20[9];
      v50 = 2819;
      v23 = (_BYTE *)sub_AD64C0(v22, v16 << (char)v21, 0);
      v24 = sub_929C50(v17, *a1[4], v23, (__int64)v51, 0, 0);
      v25 = sub_24D30A0((__int64 *)v17, 0x30u, v24, v19, (__int64)v55, 0, v48, 0);
      v26 = *a1;
      v56 = &v47;
      v45 = v25;
      v57 = 2819;
      v55[0] = "bad.descriptor";
      v27 = (__int64 **)sub_BCE3C0((__int64 *)v26[9], 0);
      v28 = sub_AD64C0((__int64)a1[5][9], -v47, 1u);
      v29 = sub_24D30A0((__int64 *)v26, 0x30u, v28, v27, (__int64)v55, 0, v53[0], 0);
      v30 = *a1;
      v31 = v29;
      v32 = sub_AA4E30((__int64)(*a1)[6]);
      v33 = sub_AE5020(v32, *(_QWORD *)(v31 + 8));
      HIBYTE(v34) = HIBYTE(v46);
      v57 = 257;
      LOBYTE(v34) = v33;
      v46 = v34;
      v35 = sub_BD2C40(80, unk_3F10A10);
      v37 = (__int64)v35;
      if ( v35 )
        sub_B4D3C0((__int64)v35, v31, v45, 0, v46, v36, 0, 0);
      (*(void (__fastcall **)(unsigned int *, __int64, _QWORD *, unsigned int *, unsigned int *))(*(_QWORD *)v30[11]
                                                                                                + 16LL))(
        v30[11],
        v37,
        v55,
        v30[7],
        v30[8]);
      v38 = *v30;
      v39 = (__int64)&(*v30)[4 * *((unsigned int *)v30 + 2)];
      if ( v38 != (unsigned int *)v39 )
      {
        v40 = v38;
        do
        {
          v41 = *((_QWORD *)v40 + 1);
          v42 = *v40;
          v40 += 4;
          sub_B99FD0(v37, v42, v41);
        }
        while ( (unsigned int *)v39 != v40 );
      }
      v16 = v47 + 1;
      result = (unsigned __int64 *)a1[3];
      v47 = v16;
    }
    while ( *result > v16 );
  }
  return result;
}
