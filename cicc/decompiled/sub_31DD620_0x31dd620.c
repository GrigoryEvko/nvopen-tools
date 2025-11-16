// Function: sub_31DD620
// Address: 0x31dd620
//
__int64 __fastcall sub_31DD620(__int64 a1, int a2, int a3)
{
  __int64 v4; // rbx
  __int64 v5; // r12
  int v6; // eax
  __int64 v7; // rdx
  char *v8; // rcx
  const char *v10; // [rsp+10h] [rbp-210h]
  __int64 v11; // [rsp+20h] [rbp-200h]
  __int64 v12; // [rsp+30h] [rbp-1F0h]
  _QWORD v13[2]; // [rsp+40h] [rbp-1E0h] BYREF
  __int128 v14; // [rsp+50h] [rbp-1D0h]
  __int64 v15; // [rsp+60h] [rbp-1C0h]
  __int128 v16; // [rsp+70h] [rbp-1B0h]
  char v17; // [rsp+90h] [rbp-190h]
  char v18; // [rsp+91h] [rbp-18Fh]
  _OWORD v19[2]; // [rsp+A0h] [rbp-180h] BYREF
  char v20; // [rsp+C0h] [rbp-160h]
  char v21; // [rsp+C1h] [rbp-15Fh]
  __int128 v22; // [rsp+D0h] [rbp-150h]
  __int16 v23; // [rsp+F0h] [rbp-130h]
  _QWORD v24[2]; // [rsp+100h] [rbp-120h] BYREF
  __int128 v25; // [rsp+110h] [rbp-110h]
  char v26; // [rsp+120h] [rbp-100h]
  char v27; // [rsp+121h] [rbp-FFh]
  __int128 v28; // [rsp+130h] [rbp-F0h]
  __int64 v29; // [rsp+150h] [rbp-D0h]
  _QWORD v30[2]; // [rsp+160h] [rbp-C0h] BYREF
  __int128 v31; // [rsp+170h] [rbp-B0h]
  char v32; // [rsp+180h] [rbp-A0h]
  char v33; // [rsp+181h] [rbp-9Fh]
  __int128 v34; // [rsp+190h] [rbp-90h]
  __int64 v35; // [rsp+1B0h] [rbp-70h]
  const char *v36[2]; // [rsp+1C0h] [rbp-60h] BYREF
  __int128 v37; // [rsp+1D0h] [rbp-50h]
  char v38; // [rsp+1E0h] [rbp-40h]
  char v39; // [rsp+1E1h] [rbp-3Fh]

  v4 = sub_31DA930(a1);
  LOWORD(v35) = 265;
  v5 = *(_QWORD *)(a1 + 216);
  *(_QWORD *)&v28 = "_set_";
  LODWORD(v34) = a3;
  LOWORD(v29) = 259;
  v23 = 265;
  LODWORD(v22) = a2;
  v18 = 1;
  *(_QWORD *)&v16 = "_";
  v17 = 3;
  v6 = sub_31DA6A0(a1);
  switch ( *(_DWORD *)(v4 + 24) )
  {
    case 0:
      v7 = 0;
      v8 = (char *)byte_3F871B3;
      break;
    case 1:
    case 3:
      v7 = 2;
      v8 = ".L";
      break;
    case 2:
    case 4:
      v7 = 1;
      v8 = "L";
      break;
    case 5:
      v7 = 2;
      v8 = "L#";
      break;
    case 6:
      v7 = 1;
      v8 = "$";
      break;
    case 7:
      v7 = 3;
      v8 = "L..";
      break;
    default:
      BUG();
  }
  LODWORD(v14) = v6;
  v13[0] = v8;
  v13[1] = v7;
  LOWORD(v15) = 2309;
  v19[1] = v16;
  *(_QWORD *)&v19[0] = v13;
  v20 = 2;
  v21 = v17;
  v27 = v23;
  v24[0] = v19;
  v24[1] = v12;
  v25 = v22;
  v26 = 2;
  v30[0] = v24;
  v31 = v28;
  v30[1] = v11;
  v32 = 2;
  v33 = v29;
  v36[0] = (const char *)v30;
  v37 = v34;
  v36[1] = v10;
  v38 = 2;
  v39 = v35;
  return sub_E6C460(v5, v36);
}
