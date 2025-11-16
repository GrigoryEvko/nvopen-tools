// Function: sub_3904470
// Address: 0x3904470
//
__int64 __fastcall sub_3904470(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5)
{
  unsigned int v8; // r12d
  __int64 v10; // rdi
  __int64 v11; // rdi
  void (*v12)(); // rax
  __int64 v13; // rdi
  __int64 v14; // [rsp+0h] [rbp-90h] BYREF
  __int64 v15; // [rsp+8h] [rbp-88h]
  unsigned int v16; // [rsp+14h] [rbp-7Ch] BYREF
  unsigned int v17; // [rsp+18h] [rbp-78h] BYREF
  unsigned int v18; // [rsp+1Ch] [rbp-74h] BYREF
  _QWORD v19[2]; // [rsp+20h] [rbp-70h] BYREF
  __int16 v20; // [rsp+30h] [rbp-60h]
  _QWORD v21[2]; // [rsp+40h] [rbp-50h] BYREF
  __int16 v22; // [rsp+50h] [rbp-40h]

  v14 = a2;
  v15 = a3;
  v8 = sub_3904260(a1, &v16, &v17, &v18);
  if ( !(_BYTE)v8 )
  {
    v10 = *(_QWORD *)(a1 + 8);
    v21[0] = "unexpected token";
    v22 = 259;
    v8 = sub_3909E20(v10, 9, v21);
    if ( (_BYTE)v8 )
    {
      v19[0] = " in '";
      v19[1] = &v14;
      v22 = 770;
      v13 = *(_QWORD *)(a1 + 8);
      v20 = 1283;
      v21[0] = v19;
      v21[1] = "' directive";
      return (unsigned int)sub_39094A0(v13, v21);
    }
    else
    {
      sub_39037E0(a1, v14, v15, 0, 0, a4, dword_452FBB0[a5]);
      v11 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
      v12 = *(void (**)())(*(_QWORD *)v11 + 216LL);
      if ( v12 != nullsub_584 )
        ((void (__fastcall *)(__int64, _QWORD, _QWORD, _QWORD, _QWORD))v12)(v11, a5, v16, v17, v18);
    }
  }
  return v8;
}
