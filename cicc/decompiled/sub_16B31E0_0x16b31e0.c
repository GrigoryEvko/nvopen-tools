// Function: sub_16B31E0
// Address: 0x16b31e0
//
__int64 __fastcall sub_16B31E0(__int64 a1, int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _BYTE *v7; // rdi
  __int64 result; // rax
  __int64 v9; // rdx
  __int64 v10; // rsi
  __int64 v11; // r12
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rdi
  _BYTE *v16; // rax
  _BYTE *v17; // rdx
  _BYTE *v18; // rax
  _BYTE *v19; // rdx
  __int64 v20; // rbx
  __int64 v21; // r13
  void (__fastcall *v22)(_BYTE *, __int64, __int64); // rax
  __int64 v23; // [rsp-8h] [rbp-60h]
  unsigned __int8 v24; // [rsp+17h] [rbp-41h] BYREF
  _BYTE v25[16]; // [rsp+18h] [rbp-40h] BYREF
  void (__fastcall *v26)(_BYTE *, _BYTE *, __int64); // [rsp+28h] [rbp-30h]
  void (__fastcall *v27)(_BYTE *, __int64); // [rsp+30h] [rbp-28h]

  v7 = (_BYTE *)(a1 + 176);
  v24 = 0;
  result = sub_16B3040((__int64)v7, a1, a3, a4, a5, a6, &v24);
  if ( !(_BYTE)result )
  {
    result = v24;
    if ( v24 )
    {
      if ( qword_4F9FB70 )
      {
        v10 = sub_16E8C20(v7, a1, v23);
        if ( qword_4F9FB70 )
        {
          qword_4F9FB78(&unk_4F9FB60, v10);
          exit(0);
        }
      }
      else
      {
        v11 = sub_16E8C20(v7, a1, v23);
        v12 = sub_1549FF0(v11, "NVIDIA", 6u);
        sub_1263B40(v12, " ");
        v13 = sub_1549FF0(v11, "NVVM", 4u);
        v14 = sub_1549FF0(v13, " version ", 9u);
        sub_1263B40(v14, "7.0.1");
        sub_1263B40(v11, "\n  ");
        v10 = (__int64)"Optimized build";
        v15 = v11;
        sub_1263B40(v11, "Optimized build");
        v16 = *(_BYTE **)(v11 + 24);
        if ( (unsigned __int64)v16 >= *(_QWORD *)(v11 + 16) )
        {
          v10 = 10;
          v15 = v11;
          sub_16E7DE0(v11, 10);
        }
        else
        {
          v17 = v16 + 1;
          *(_QWORD *)(v11 + 24) = v16 + 1;
          *v16 = 10;
        }
        if ( !qword_4F9FB48 )
          goto LABEL_19;
        v7 = (_BYTE *)sub_16E8C20(v15, v10, v17);
        v18 = (_BYTE *)*((_QWORD *)v7 + 3);
        if ( (unsigned __int64)v18 >= *((_QWORD *)v7 + 2) )
        {
          v10 = 10;
          sub_16E7DE0(v7, 10);
        }
        else
        {
          v19 = v18 + 1;
          *((_QWORD *)v7 + 3) = v18 + 1;
          *v18 = 10;
        }
        v20 = *(_QWORD *)qword_4F9FB48;
        v21 = *(_QWORD *)(qword_4F9FB48 + 8);
        if ( *(_QWORD *)qword_4F9FB48 == v21 )
LABEL_19:
          exit(0);
        while ( 1 )
        {
          v26 = 0;
          v22 = *(void (__fastcall **)(_BYTE *, __int64, __int64))(v20 + 16);
          if ( v22 )
          {
            v10 = v20;
            v7 = v25;
            v22(v25, v20, 2);
            v27 = *(void (__fastcall **)(_BYTE *, __int64))(v20 + 24);
            v26 = *(void (__fastcall **)(_BYTE *, _BYTE *, __int64))(v20 + 16);
          }
          v10 = sub_16E8C20(v7, v10, v19);
          if ( !v26 )
            break;
          v7 = v25;
          v27(v25, v10);
          if ( v26 )
          {
            v10 = (__int64)v25;
            v7 = v25;
            v26(v25, v25, 3);
          }
          v20 += 32;
          if ( v21 == v20 )
            goto LABEL_19;
        }
      }
      sub_4263D6(v7, v10, v9);
    }
    *(_DWORD *)(a1 + 16) = a2;
  }
  return result;
}
