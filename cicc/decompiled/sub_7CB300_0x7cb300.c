// Function: sub_7CB300
// Address: 0x7cb300
//
__int64 __fastcall sub_7CB300(const char *a1, int a2, int a3, char a4, __int64 a5)
{
  _BYTE *v7; // rdi
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  unsigned __int64 v12; // rdi
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 result; // rax
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // r9
  bool v26; // [rsp+Bh] [rbp-E5h]
  _BYTE *v27; // [rsp+10h] [rbp-E0h] BYREF
  __int64 v28; // [rsp+18h] [rbp-D8h]
  __int64 v29; // [rsp+20h] [rbp-D0h]
  _BYTE v30[32]; // [rsp+30h] [rbp-C0h] BYREF
  _DWORD v31[8]; // [rsp+50h] [rbp-A0h] BYREF
  int v32; // [rsp+70h] [rbp-80h]
  __int64 v33; // [rsp+78h] [rbp-78h]
  int v34; // [rsp+80h] [rbp-70h]
  int v35; // [rsp+84h] [rbp-6Ch]
  __int64 v36; // [rsp+88h] [rbp-68h]
  int v37; // [rsp+90h] [rbp-60h]
  int v38; // [rsp+94h] [rbp-5Ch]
  __int64 v39; // [rsp+98h] [rbp-58h]
  __int64 v40; // [rsp+A0h] [rbp-50h]
  _BYTE *v41; // [rsp+A8h] [rbp-48h] BYREF
  _QWORD v42[8]; // [rsp+B0h] [rbp-40h] BYREF

  v26 = (qword_4F061C0[7] & 4) != 0;
  v32 = dword_4F17F78;
  v33 = qword_4F06498;
  v34 = unk_4F061F8;
  v35 = dword_4F17FA0;
  v36 = qword_4F06490;
  v37 = dword_4F04D80;
  v38 = dword_4D03D1C;
  v39 = qword_4F17F70;
  qword_4F17F70 = a5;
  dword_4F17FA0 = 0;
  v27 = 0;
  v42[0] = unk_4D03BE0;
  v40 = unk_4D03BE0;
  v42[1] = &v41;
  unk_4D03BE0 = v42;
  v28 = 0;
  v29 = 0;
  v41 = qword_4F06460;
  *((_BYTE *)qword_4F061C0 + 56) = (4 * (a4 & 1)) | qword_4F061C0[7] & 0xFB;
  v31[0] = (_DWORD)&loc_1000200;
  unk_4F061F8 = 1;
  v27 = (_BYTE *)sub_823970(0);
  v28 = 0;
  sub_7CB0D0((__int64 *)&v27, (__int64)v30, a1, v31, 4);
  dword_4F17F78 = 1;
  qword_4F06460 = v27;
  qword_4F06498 = v27;
  dword_4F04D80 = 1;
  qword_4F06490 = &v27[v29 - 1];
  dword_4D03D1C = a3;
  sub_7ADF70((__int64)v30, 0);
  sub_7ABA40();
  if ( !a2 )
  {
    sub_7ADF70((__int64)v31, 0);
    v7 = v31;
    sub_7AE360((__int64)v31);
    goto LABEL_3;
  }
  do
  {
    v7 = v30;
    sub_7AE360((__int64)v30);
LABEL_3:
    *(_QWORD *)&dword_4F063F8 = a5;
    qword_4F063F0 = a5;
  }
  while ( (unsigned __int16)sub_7B8B50((unsigned __int64)v7, 0, v8, v9, v10, v11) != 10 );
  v12 = (unsigned __int64)v27;
  sub_823A00(v27, v28);
  qword_4F06460 = v41;
  unk_4D03BE0 = v40;
  qword_4F17F70 = v39;
  dword_4D03D1C = v38;
  dword_4F04D80 = v37;
  qword_4F06490 = v36;
  dword_4F17FA0 = v35;
  unk_4F061F8 = v34;
  qword_4F06498 = v33;
  dword_4F17F78 = v32;
  sub_7AB810();
  sub_7B8B50(v12, (unsigned int *)&qword_4F06498, v13, v14, v15, v16);
  sub_7BC000((unsigned __int64)v31, (__int64)&qword_4F06498, v22, v23, v24, v25);
  sub_7BC000((unsigned __int64)v30, (__int64)&qword_4F06498, v17, v18, v19, v20);
  result = qword_4F061C0[7] & 0xFB | (4 * (unsigned int)v26);
  *((_BYTE *)qword_4F061C0 + 56) = qword_4F061C0[7] & 0xFB | (4 * v26);
  return result;
}
