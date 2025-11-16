// Function: sub_3987310
// Address: 0x3987310
//
__int64 __fastcall sub_3987310(__int64 a1, unsigned __int8 a2, char *a3)
{
  __int64 (__fastcall ***v3)(_QWORD, _QWORD, _QWORD *, __int64, __int64, __int64, const char *, __int64, const char *, __int64, char *, char *, int); // r13
  __int64 (__fastcall *v4)(_QWORD, _QWORD, _QWORD *, __int64, __int64, __int64, const char *, __int64, const char *, __int64, char *, char *, int); // r14
  const char *v6; // rax
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  bool v10; // zf
  __int64 v11; // rdx
  char **v12; // rax
  char v13; // dl
  __int64 v15; // rdx
  const char *v16; // [rsp+0h] [rbp-80h] BYREF
  __int64 v17; // [rsp+8h] [rbp-78h]
  const char *v18; // [rsp+10h] [rbp-70h] BYREF
  __int64 v19; // [rsp+18h] [rbp-68h]
  char *v20; // [rsp+20h] [rbp-60h] BYREF
  char *v21; // [rsp+28h] [rbp-58h]
  int v22; // [rsp+30h] [rbp-50h]
  _QWORD v23[2]; // [rsp+40h] [rbp-40h] BYREF
  __int16 v24; // [rsp+50h] [rbp-30h]

  v3 = *(__int64 (__fastcall ****)(_QWORD, _QWORD, _QWORD *, __int64, __int64, __int64, const char *, __int64, const char *, __int64, char *, char *, int))(a1 + 88);
  v4 = **v3;
  if ( a3 )
  {
    v6 = sub_14E3970(a2);
    v10 = *a3 == 0;
    v16 = v6;
    v17 = v11;
    if ( v10 )
    {
      v12 = (char **)" ";
      v20 = " ";
      LOWORD(v22) = 259;
      v13 = 3;
    }
    else
    {
      v7 = 771;
      v20 = a3;
      v13 = 2;
      v21 = " ";
      v12 = &v20;
      LOWORD(v22) = 771;
    }
    v23[0] = v12;
    v23[1] = &v16;
    LOBYTE(v24) = v13;
    HIBYTE(v24) = 5;
  }
  else
  {
    v18 = sub_14E3970(a2);
    v24 = 261;
    v19 = v15;
    v23[0] = &v18;
  }
  return v4(v3, a2, v23, v7, v8, v9, v16, v17, v18, v19, v20, v21, v22);
}
