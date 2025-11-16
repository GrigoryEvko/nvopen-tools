// Function: sub_106E0D0
// Address: 0x106e0d0
//
__int64 *__fastcall sub_106E0D0(__int64 *a1, _DWORD *a2, __int64 a3, __int64 a4)
{
  int v7; // eax
  __int64 v8; // rax
  bool v9; // zf
  _BOOL4 v10; // r15d
  __int64 v11; // rax
  __int64 v12; // rbx
  __int64 v14; // rax
  __int64 v15; // rdi
  __int64 v16; // rax
  __int64 v17; // [rsp+0h] [rbp-40h] BYREF
  _QWORD v18[7]; // [rsp+8h] [rbp-38h] BYREF

  (*(void (__fastcall **)(__int64 *))(*(_QWORD *)a2 + 40LL))(&v17);
  v7 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v17 + 16LL))(v17);
  switch ( v7 )
  {
    case 3:
      v8 = v17;
      v9 = a2[2] == 1;
      v17 = 0;
      v18[0] = v8;
      v10 = v9;
      v11 = sub_22077B0(224);
      v12 = v11;
      if ( v11 )
        sub_124C750(v11, v18, a3, a4, v10);
      if ( v18[0] )
        (*(void (__fastcall **)(_QWORD))(*(_QWORD *)v18[0] + 8LL))(v18[0]);
      *a1 = v12;
      break;
    case 7:
      v16 = v17;
      v17 = 0;
      v18[0] = v16;
      sub_107C150(a1, v18, a3, a4);
      v15 = v18[0];
      if ( v18[0] )
LABEL_14:
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v15 + 8LL))(v15);
      break;
    case 1:
      v14 = v17;
      v17 = 0;
      v18[0] = v14;
      sub_108AD50(a1, v18, a3, a4);
      v15 = v18[0];
      if ( v18[0] )
        goto LABEL_14;
      break;
    default:
      sub_C64ED0("dwo only supported with COFF, ELF, and Wasm", 1u);
  }
  if ( v17 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v17 + 8LL))(v17);
  return a1;
}
