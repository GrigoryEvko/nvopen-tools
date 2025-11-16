// Function: sub_396FFC0
// Address: 0x396ffc0
//
__int64 __fastcall sub_396FFC0(__int64 a1, int a2, int a3)
{
  __int64 v3; // rbx
  __int64 v4; // r15
  __int64 v5; // rax
  char *v6; // rdx
  _QWORD v8[2]; // [rsp+0h] [rbp-180h] BYREF
  __int64 v9; // [rsp+10h] [rbp-170h]
  _QWORD v10[2]; // [rsp+30h] [rbp-150h] BYREF
  __int64 v11; // [rsp+40h] [rbp-140h]
  char *v12; // [rsp+50h] [rbp-130h]
  char v13; // [rsp+60h] [rbp-120h]
  char v14; // [rsp+61h] [rbp-11Fh]
  _QWORD v15[2]; // [rsp+70h] [rbp-110h] BYREF
  char v16; // [rsp+80h] [rbp-100h]
  char v17; // [rsp+81h] [rbp-FFh]
  __int64 v18; // [rsp+90h] [rbp-F0h]
  __int16 v19; // [rsp+A0h] [rbp-E0h]
  _QWORD v20[2]; // [rsp+B0h] [rbp-D0h] BYREF
  char v21; // [rsp+C0h] [rbp-C0h]
  char v22; // [rsp+C1h] [rbp-BFh]
  __int128 v23; // [rsp+D0h] [rbp-B0h]
  __int64 v24; // [rsp+E0h] [rbp-A0h]
  _QWORD v25[2]; // [rsp+F0h] [rbp-90h] BYREF
  char v26; // [rsp+100h] [rbp-80h]
  char v27; // [rsp+101h] [rbp-7Fh]
  __int128 v28; // [rsp+110h] [rbp-70h]
  __int64 v29; // [rsp+120h] [rbp-60h]
  _QWORD v30[2]; // [rsp+130h] [rbp-50h] BYREF
  char v31; // [rsp+140h] [rbp-40h]
  char v32; // [rsp+141h] [rbp-3Fh]

  LODWORD(v28) = a3;
  v3 = sub_396DDB0(a1);
  LOWORD(v29) = 265;
  *(_QWORD *)&v23 = "_set_";
  v4 = *(_QWORD *)(a1 + 248);
  LOWORD(v24) = 259;
  v19 = 265;
  LODWORD(v18) = a2;
  v14 = 1;
  v12 = "_";
  v13 = 3;
  LODWORD(v9) = sub_396DD70(a1);
  switch ( *(_DWORD *)(v3 + 16) )
  {
    case 0:
      v5 = 0;
      v6 = (char *)byte_3F871B3;
      break;
    case 1:
    case 3:
      v5 = 2;
      v6 = ".L";
      break;
    case 2:
    case 4:
      v5 = 1;
      v6 = "L";
      break;
    case 5:
      v5 = 1;
      v6 = "$";
      break;
  }
  v8[1] = v5;
  v10[0] = v8;
  v8[0] = v6;
  v10[1] = v9;
  LOWORD(v11) = 2309;
  v15[1] = v12;
  v15[0] = v10;
  v16 = 2;
  v17 = v13;
  v22 = v19;
  v20[0] = v15;
  v20[1] = v18;
  v21 = 2;
  v27 = v24;
  v25[0] = v20;
  v25[1] = v23;
  v26 = 2;
  v30[0] = v25;
  v30[1] = v28;
  v31 = 2;
  v32 = v29;
  return sub_38BF510(v4, (__int64)v30);
}
