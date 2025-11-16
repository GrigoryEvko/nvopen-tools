// Function: sub_EC8D60
// Address: 0xec8d60
//
__int64 __fastcall sub_EC8D60(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4, unsigned int a5)
{
  __int64 v9; // rdi
  _DWORD *v10; // rdi
  __int64 v11; // rax
  __int64 v12; // r9
  unsigned int v13; // r10d
  __int64 v14; // rdi
  void (*v15)(); // rax
  __int64 v17; // rdi
  unsigned int v18; // [rsp+14h] [rbp-ACh] BYREF
  unsigned int v19; // [rsp+18h] [rbp-A8h] BYREF
  unsigned int v20; // [rsp+1Ch] [rbp-A4h] BYREF
  __int128 v21; // [rsp+20h] [rbp-A0h] BYREF
  _QWORD v22[4]; // [rsp+30h] [rbp-90h] BYREF
  __int16 v23; // [rsp+50h] [rbp-70h]
  _QWORD v24[4]; // [rsp+60h] [rbp-60h] BYREF
  __int16 v25; // [rsp+80h] [rbp-40h]

  if ( (unsigned __int8)sub_EC83C0(a1, &v18, &v19, "OS") )
    return 1;
  if ( (unsigned __int8)sub_EC73D0(a1, &v20) )
    return 1;
  v9 = *(_QWORD *)(a1 + 8);
  v21 = 0;
  v10 = *(_DWORD **)((*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v9 + 40LL))(v9) + 8);
  if ( *v10 == 2 && sub_EC5140((__int64)v10) && (unsigned __int8)sub_EC8740(a1, (__int64)&v21) )
  {
    return 1;
  }
  else if ( (unsigned __int8)sub_ECE000(*(_QWORD *)(a1 + 8)) )
  {
    v22[2] = a2;
    v22[0] = " in '";
    v23 = 1283;
    v17 = *(_QWORD *)(a1 + 8);
    v24[0] = v22;
    v22[3] = a3;
    v24[2] = "' directive";
    v25 = 770;
    return (unsigned int)sub_ECD7F0(v17, v24);
  }
  else
  {
    sub_EC6AF0(a1, a2, a3, 0, 0, a4, dword_3F864E0[a5]);
    v11 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
    v13 = 0;
    v14 = v11;
    v15 = *(void (**)())(*(_QWORD *)v11 + 248LL);
    if ( v15 != nullsub_102 )
    {
      ((void (__fastcall *)(__int64, _QWORD, _QWORD, _QWORD, _QWORD, __int64, _QWORD, _QWORD))v15)(
        v14,
        a5,
        v18,
        v19,
        v20,
        v12,
        v21,
        *((_QWORD *)&v21 + 1));
      return 0;
    }
  }
  return v13;
}
