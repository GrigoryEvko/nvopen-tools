// Function: sub_7CB620
// Address: 0x7cb620
//
_QWORD *__fastcall sub_7CB620(const char *a1, __int64 a2, __int64 *a3)
{
  _BYTE **v5; // rdi
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v15; // [rsp+0h] [rbp-E0h]
  const char *v16; // [rsp+8h] [rbp-D8h]
  char v17; // [rsp+1Bh] [rbp-C5h] BYREF
  int v18; // [rsp+1Ch] [rbp-C4h] BYREF
  _BYTE *v19; // [rsp+20h] [rbp-C0h] BYREF
  __int64 v20; // [rsp+28h] [rbp-B8h]
  __int64 v21; // [rsp+30h] [rbp-B0h]
  _BYTE v22[32]; // [rsp+40h] [rbp-A0h] BYREF
  int v23; // [rsp+60h] [rbp-80h]
  __int64 v24; // [rsp+68h] [rbp-78h]
  int v25; // [rsp+70h] [rbp-70h]
  int v26; // [rsp+74h] [rbp-6Ch]
  __int64 v27; // [rsp+78h] [rbp-68h]
  int v28; // [rsp+80h] [rbp-60h]
  int v29; // [rsp+84h] [rbp-5Ch]
  __int64 v30; // [rsp+88h] [rbp-58h]
  __int64 v31; // [rsp+90h] [rbp-50h]
  _BYTE *v32; // [rsp+98h] [rbp-48h] BYREF
  _QWORD v33[8]; // [rsp+A0h] [rbp-40h] BYREF

  v16 = qword_4F06410;
  v15 = qword_4F06408;
  v23 = dword_4F17F78;
  v24 = qword_4F06498;
  v25 = unk_4F061F8;
  v26 = dword_4F17FA0;
  v27 = qword_4F06490;
  v28 = dword_4F04D80;
  v29 = dword_4D03D1C;
  v30 = qword_4F17F70;
  v31 = unk_4D03BE0;
  v33[0] = unk_4D03BE0;
  v33[1] = &v32;
  unk_4D03BE0 = v33;
  v32 = qword_4F06460;
  qword_4F17F70 = *a3;
  sub_7ADF70((__int64)v22, 0);
  sub_7AE360((__int64)v22);
  sub_7B8190();
  dword_4F17FA0 = 0;
  v18 = (int)&loc_1000200;
  unk_4F061F8 = 1;
  v19 = 0;
  v20 = 0;
  v21 = 0;
  v5 = &v19;
  v19 = (_BYTE *)sub_823970(0);
  v20 = 0;
  sub_7CB0D0((__int64 *)&v19, (__int64)&v17, a1, &v18, 4);
  dword_4F17F78 = 1;
  qword_4F06460 = v19;
  qword_4F06498 = v19;
  qword_4F06490 = &v19[v21 - 1];
  dword_4F04D80 = 1;
  dword_4D03D1C = 0;
  sub_7ABA40();
  while ( 1 )
  {
    *(_QWORD *)&dword_4F063F8 = *a3;
    qword_4F063F0 = *a3;
    if ( (unsigned __int16)sub_7B8B50((unsigned __int64)v5, (unsigned int *)&v17, v6, v7, v8, v9) == 10 )
      break;
    v5 = (_BYTE **)a2;
    sub_7AE360(a2);
  }
  sub_823A00(v19, v20);
  qword_4F06460 = v32;
  unk_4D03BE0 = v31;
  qword_4F17F70 = v30;
  dword_4D03D1C = v29;
  dword_4F04D80 = v28;
  qword_4F06490 = v27;
  dword_4F17FA0 = v26;
  unk_4F061F8 = v25;
  qword_4F06498 = v24;
  dword_4F17F78 = v23;
  sub_7AB810(v22, 1);
  sub_7BBF80((unsigned __int64)v22, (unsigned int *)1, v10, v11, v12, v13);
  qword_4F06408 = v15;
  qword_4F06410 = v16;
  return sub_7B8260();
}
