// Function: sub_23B36B0
// Address: 0x23b36b0
//
__int64 *__fastcall sub_23B36B0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, char a7)
{
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rax
  _QWORD v13[2]; // [rsp+10h] [rbp-240h] BYREF
  _BYTE *v14; // [rsp+20h] [rbp-230h] BYREF
  __int64 v15; // [rsp+28h] [rbp-228h]
  __int64 v16; // [rsp+30h] [rbp-220h]
  _BYTE v17[24]; // [rsp+38h] [rbp-218h] BYREF
  _QWORD v18[8]; // [rsp+50h] [rbp-200h] BYREF
  _QWORD v19[10]; // [rsp+90h] [rbp-1C0h] BYREF
  const char *v20; // [rsp+E0h] [rbp-170h] BYREF
  __int64 v21; // [rsp+E8h] [rbp-168h]
  _QWORD *v22; // [rsp+F0h] [rbp-160h]
  __int64 v23; // [rsp+F8h] [rbp-158h]
  __int64 v24; // [rsp+100h] [rbp-150h]
  __int64 v25; // [rsp+108h] [rbp-148h] BYREF
  _QWORD *v26; // [rsp+110h] [rbp-140h]
  _QWORD v27[2]; // [rsp+118h] [rbp-138h] BYREF
  _QWORD v28[3]; // [rsp+128h] [rbp-128h] BYREF
  __int64 *v29; // [rsp+140h] [rbp-110h] BYREF
  __int64 v30; // [rsp+148h] [rbp-108h]
  unsigned __int64 v31; // [rsp+150h] [rbp-100h]
  __int64 v32; // [rsp+158h] [rbp-F8h] BYREF
  char v33; // [rsp+160h] [rbp-F0h]
  _QWORD v34[2]; // [rsp+168h] [rbp-E8h] BYREF
  _QWORD v35[2]; // [rsp+178h] [rbp-D8h] BYREF
  _QWORD v36[25]; // [rsp+188h] [rbp-C8h] BYREF

  v29 = (__int64 *)"{0}/{1}";
  v31 = (unsigned __int64)v36;
  v25 = 0x100000000LL;
  v13[1] = a3;
  v34[0] = &unk_49DB108;
  v34[1] = &a7;
  v13[0] = a2;
  v35[0] = &unk_4A16028;
  v35[1] = &qword_4FDEC80;
  v36[0] = v35;
  v36[1] = v34;
  v30 = 7;
  v32 = 2;
  v33 = 1;
  v14 = v17;
  v15 = 0;
  v16 = 20;
  v21 = 2;
  v22 = 0;
  v23 = 0;
  v24 = 0;
  v20 = (const char *)&unk_49DD288;
  v26 = &v14;
  sub_CB5980((__int64)&v20, 0, 0, 0);
  sub_CB6840((__int64)&v20, (__int64)&v29);
  v20 = (const char *)&unk_49DD388;
  sub_CB5840((__int64)&v20);
  if ( !byte_4FDE2E8 && (unsigned int)sub_2207590((__int64)&byte_4FDE2E8) )
  {
    sub_C86E60((char *)&qword_4FDE300, qword_4FDF108, qword_4FDF110, 0, 0);
    __cxa_atexit((void (*)(void *))sub_BC5B10, &qword_4FDE300, &qword_4A427C0);
    sub_2207640((__int64)&byte_4FDE2E8);
  }
  if ( (byte_4FDE320 & 1) != 0 )
  {
    sub_23B0820(a1, "Unable to find dot executable.");
  }
  else
  {
    LOBYTE(v31) = 0;
    v19[0] = qword_4FDF108;
    v19[1] = qword_4FDF110;
    v19[2] = "-Tpdf";
    v19[4] = "-o";
    v19[6] = v14;
    v19[7] = v15;
    v19[8] = a4;
    v19[3] = 5;
    v19[5] = 2;
    v19[9] = a5;
    if ( (int)sub_C881F0((_BYTE *)qword_4FDE300, qword_4FDE308, v19, 5, 0, 0, v29, v30, 0, 0, 0, 0, 0, 0) < 0 )
    {
      sub_23B0820(a1, "Error executing system dot.");
    }
    else
    {
      v20 = "  <a href=\"{0}\" target=\"_blank\">{1}</a><br/>\n";
      v22 = v28;
      v26 = v13;
      v25 = (__int64)&unk_49DB108;
      v27[0] = &unk_49DB108;
      v27[1] = &a7;
      v28[0] = v27;
      v28[1] = &v25;
      v18[5] = 0x100000000LL;
      v21 = 45;
      LOBYTE(v24) = 1;
      v18[0] = &unk_49DD288;
      v23 = 2;
      v29 = &v32;
      v30 = 0;
      v31 = 200;
      v18[1] = 2;
      memset(&v18[2], 0, 24);
      v18[6] = &v29;
      sub_CB5980((__int64)v18, 0, 0, 0);
      sub_CB6840((__int64)v18, (__int64)&v20);
      v18[0] = &unk_49DD388;
      sub_CB5840((__int64)v18);
      v11 = v30;
      if ( v30 + 1 > v31 )
      {
        sub_C8D290((__int64)&v29, &v32, v30 + 1, 1u, v9, v10);
        v11 = v30;
      }
      *((_BYTE *)v29 + v11) = 0;
      sub_23B0820(a1, (const char *)v29);
      if ( v29 != &v32 )
        _libc_free((unsigned __int64)v29);
    }
  }
  if ( v14 != v17 )
    _libc_free((unsigned __int64)v14);
  return a1;
}
