// Function: sub_A03600
// Address: 0xa03600
//
void __fastcall sub_A03600(_BYTE *a1, _BYTE *a2, __int64 a3)
{
  _BYTE *v3; // r13
  int v4; // r12d
  _BYTE *v5; // rbx
  void (__fastcall *v6)(_BYTE *, _BYTE *, __int64); // rax
  void (__fastcall *v7)(_BYTE *, _BYTE *, __int64); // rax
  bool v8; // zf
  int v9; // [rsp-74h] [rbp-74h] BYREF
  _BYTE *v10; // [rsp-70h] [rbp-70h] BYREF
  _BYTE v11[16]; // [rsp-68h] [rbp-68h] BYREF
  void (__fastcall *v12)(_BYTE *, _BYTE *, __int64); // [rsp-58h] [rbp-58h]
  __int64 v13; // [rsp-50h] [rbp-50h]
  _BYTE v14[16]; // [rsp-48h] [rbp-48h] BYREF
  void (__fastcall *v15)(_BYTE *, _BYTE *, __int64); // [rsp-38h] [rbp-38h]
  __int64 v16; // [rsp-30h] [rbp-30h]

  if ( a1[360] )
  {
    v3 = a2;
    v4 = a3;
    v5 = a1;
    v6 = (void (__fastcall *)(_BYTE *, _BYTE *, __int64))*((_QWORD *)a1 + 39);
    v12 = 0;
    if ( v6 )
    {
      a2 = a1 + 296;
      a1 = v11;
      v6(v11, a2, 2);
      v13 = *((_QWORD *)v5 + 40);
      v12 = (void (__fastcall *)(_BYTE *, _BYTE *, __int64))*((_QWORD *)v5 + 39);
    }
    v7 = (void (__fastcall *)(_BYTE *, _BYTE *, __int64))*((_QWORD *)v5 + 35);
    v15 = 0;
    if ( v7 )
    {
      a2 = v5 + 264;
      a1 = v14;
      v7(v14, v5 + 264, 2);
      v16 = *((_QWORD *)v5 + 36);
      v15 = (void (__fastcall *)(_BYTE *, _BYTE *, __int64))*((_QWORD *)v5 + 35);
    }
    v8 = *((_QWORD *)v5 + 43) == 0;
    v10 = v3;
    v9 = v4;
    if ( v8 )
      sub_4263D6(a1, a2, a3);
    (*((void (__fastcall **)(_BYTE *, _BYTE **, int *, _BYTE *, _BYTE *))v5 + 44))(v5 + 328, &v10, &v9, v14, v11);
    if ( v15 )
      v15(v14, v14, 3);
    if ( v12 )
      v12(v11, v11, 3);
  }
}
