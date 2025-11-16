// Function: sub_39B2B20
// Address: 0x39b2b20
//
__int64 __fastcall sub_39B2B20(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, char a5)
{
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rdi
  __int64 v12; // r13
  __int64 (__fastcall *v13)(_QWORD, __int64, __int64); // rax
  __int64 v14; // rax
  __int64 v15; // r8
  __int64 (__fastcall *v16)(__int64, __int64, __int64, __int64); // rax
  _DWORD *v17; // rax
  char v18; // dl
  __int64 v19; // r11
  __int64 (__fastcall *v20)(__int64, __int64 *); // rax
  __int64 v21; // rsi
  void (__fastcall *v22)(__int64, __int64, _QWORD); // rbx
  __int64 v23; // rax
  __int64 v25; // [rsp+0h] [rbp-90h]
  _DWORD *v26; // [rsp+8h] [rbp-88h]
  unsigned __int8 v27; // [rsp+1Ch] [rbp-74h]
  __int64 v28; // [rsp+20h] [rbp-70h]
  unsigned __int8 v29; // [rsp+20h] [rbp-70h]
  __int64 v30; // [rsp+28h] [rbp-68h]
  char v31; // [rsp+3Fh] [rbp-51h] BYREF
  __int64 v32; // [rsp+40h] [rbp-50h] BYREF
  _DWORD *v33; // [rsp+48h] [rbp-48h] BYREF
  __int64 v34; // [rsp+50h] [rbp-40h] BYREF
  __int64 v35[7]; // [rsp+58h] [rbp-38h] BYREF

  v31 = 1;
  v9 = sub_39B13D0((__int64 *)a1, a2, a5, (bool *)&v31, a4, 0);
  *a3 = v9;
  if ( !v9 )
    goto LABEL_19;
  v10 = v9;
  if ( (*(_BYTE *)(a1 + 840) & 0x40) != 0 )
  {
    *(_BYTE *)(v9 + 1162) = 0;
    v10 = *a3;
  }
  v11 = *(_QWORD *)(a1 + 8);
  v12 = *(_QWORD *)(a1 + 616);
  v30 = *(_QWORD *)(a1 + 632);
  v13 = *(__int64 (__fastcall **)(_QWORD, __int64, __int64))(v11 + 136);
  if ( v13 )
  {
    v14 = v13(*(_QWORD *)(a1 + 624), v12, v10);
    v11 = *(_QWORD *)(a1 + 8);
    v15 = v14;
  }
  else
  {
    v15 = 0;
  }
  v16 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(v11 + 96);
  v28 = v15;
  if ( v16 && (v17 = (_DWORD *)v16(v11, v30, v12, a1 + 840), LOBYTE(v12) = v17 == 0 || v28 == 0, !(_BYTE)v12) )
  {
    v18 = *(_BYTE *)(a1 + 841);
    v19 = *(_QWORD *)(a1 + 8);
    v35[0] = v28;
    v26 = v17;
    v25 = v19;
    v29 = v18 & 1;
    v27 = (*(_BYTE *)(a1 + 840) & 2) != 0;
    sub_390A0A0((__int64)&v34, v17, a4);
    v33 = v26;
    v32 = sub_39B29F0(v25, a1 + 472, *a3, (__int64)&v33, (__int64)&v34, (__int64)v35, v30, v27, v29, 1u);
    if ( v33 )
      (*(void (__fastcall **)(_DWORD *))(*(_QWORD *)v33 + 8LL))(v33);
    if ( v34 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v34 + 8LL))(v34);
    if ( v35[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v35[0] + 8LL))(v35[0]);
    v20 = *(__int64 (__fastcall **)(__int64, __int64 *))(*(_QWORD *)(a1 + 8) + 112LL);
    if ( v20 && (v21 = v20(a1, &v32)) != 0 )
    {
      (*(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)a2 + 16LL))(a2, v21, 0);
      v22 = *(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)a2 + 16LL);
      v23 = sub_1E2D0B0();
      v22(a2, v23, 0);
    }
    else
    {
      LODWORD(v12) = 1;
    }
    if ( v32 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v32 + 48LL))(v32);
  }
  else
  {
LABEL_19:
    LODWORD(v12) = 1;
  }
  return (unsigned int)v12;
}
