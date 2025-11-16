// Function: sub_2E3A130
// Address: 0x2e3a130
//
__int64 *__fastcall sub_2E3A130(__int64 *a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // r12
  int v4; // eax
  bool v5; // zf
  __int128 v6; // rax
  char *v8; // [rsp+0h] [rbp-110h] BYREF
  int v9; // [rsp+10h] [rbp-100h]
  __int16 v10; // [rsp+20h] [rbp-F0h]
  _QWORD v11[4]; // [rsp+30h] [rbp-E0h] BYREF
  __int16 v12; // [rsp+50h] [rbp-C0h]
  _QWORD v13[2]; // [rsp+60h] [rbp-B0h] BYREF
  __int128 v14; // [rsp+70h] [rbp-A0h]
  __int64 v15; // [rsp+80h] [rbp-90h]
  char *v16; // [rsp+90h] [rbp-80h]
  __int64 v17; // [rsp+98h] [rbp-78h]
  char v18; // [rsp+B0h] [rbp-60h]
  char v19; // [rsp+B1h] [rbp-5Fh]
  __int128 v20; // [rsp+C0h] [rbp-50h] BYREF
  char *v21; // [rsp+D0h] [rbp-40h]
  __int64 v22; // [rsp+D8h] [rbp-38h]
  __int64 v23; // [rsp+E0h] [rbp-30h]

  v4 = *(_DWORD *)(a2 + 24);
  v5 = *(_QWORD *)(a2 + 16) == 0;
  v8 = "BB";
  v10 = 2563;
  v9 = v4;
  if ( v5 )
  {
    sub_CA0F50(a1, (void **)&v8);
  }
  else
  {
    v19 = 1;
    v16 = "]";
    v18 = 3;
    *(_QWORD *)&v6 = sub_2E31BC0(a2);
    v11[1] = v3;
    v11[0] = &v8;
    v11[2] = "[";
    v12 = 770;
    v14 = v6;
    v13[0] = v11;
    v13[1] = v2;
    LOWORD(v15) = 1282;
    v20 = (unsigned __int64)v13;
    v21 = "]";
    v22 = v17;
    LOWORD(v23) = 770;
    sub_CA0F50(a1, (void **)&v20);
  }
  return a1;
}
