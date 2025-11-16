// Function: sub_3267170
// Address: 0x3267170
//
__int64 __fastcall sub_3267170(__int64 *a1, __int64 a2)
{
  _QWORD *v3; // rax
  __int64 v4; // r12
  __int64 v5; // r13
  int v6; // edx
  __int64 v7; // r14
  __int64 v8; // r15
  __int64 v9; // r10
  __int64 v10; // r11
  int v11; // ecx
  __int64 v12; // rcx
  __int64 result; // rax
  __int64 v14; // rsi
  __int64 v15; // rcx
  int v16; // r8d
  __int64 v17; // rsi
  __int64 v18; // r10
  __int64 v19; // rcx
  int v20; // r8d
  __int128 v21; // [rsp-30h] [rbp-A0h]
  __int128 v22; // [rsp-20h] [rbp-90h]
  __int128 v23; // [rsp-20h] [rbp-90h]
  __int128 v24; // [rsp-10h] [rbp-80h]
  __int128 v25; // [rsp-10h] [rbp-80h]
  int v26; // [rsp+0h] [rbp-70h]
  int v27; // [rsp+8h] [rbp-68h]
  int v28; // [rsp+8h] [rbp-68h]
  __int64 v29; // [rsp+10h] [rbp-60h]
  int v30; // [rsp+10h] [rbp-60h]
  __int64 v31; // [rsp+18h] [rbp-58h]
  int v32; // [rsp+20h] [rbp-50h]
  __int64 v33; // [rsp+20h] [rbp-50h]
  __int64 v34; // [rsp+28h] [rbp-48h]
  __int64 v35; // [rsp+28h] [rbp-48h]
  __int64 v36; // [rsp+30h] [rbp-40h] BYREF
  int v37; // [rsp+38h] [rbp-38h]

  v3 = *(_QWORD **)(a2 + 40);
  v4 = *v3;
  v5 = v3[1];
  v6 = *(_DWORD *)(*v3 + 24LL);
  v7 = v3[5];
  v8 = v3[6];
  v9 = v3[10];
  v10 = v3[11];
  v11 = *(_DWORD *)(v7 + 24);
  if ( v6 != 35 && v6 != 11 || v11 == 35 || v11 == 11 )
  {
    v12 = v3[10];
    result = 0;
    if ( *(_DWORD *)(v12 + 24) == 67 )
    {
      v17 = *(_QWORD *)(a2 + 80);
      v18 = *a1;
      v19 = *(_QWORD *)(a2 + 48);
      v20 = *(_DWORD *)(a2 + 68);
      v36 = v17;
      if ( v17 )
      {
        v28 = v19;
        v30 = v20;
        v32 = v18;
        sub_B96E90((__int64)&v36, v17, 1);
        LODWORD(v19) = v28;
        v20 = v30;
        LODWORD(v18) = v32;
      }
      *((_QWORD *)&v25 + 1) = v8;
      *(_QWORD *)&v25 = v7;
      *((_QWORD *)&v23 + 1) = v5;
      *(_QWORD *)&v23 = v4;
      v37 = *(_DWORD *)(a2 + 72);
      result = sub_3411F20(v18, 68, (unsigned int)&v36, v19, v20, (unsigned int)&v36, v23, v25);
      if ( v36 )
      {
        v33 = result;
        sub_B91220((__int64)&v36, v36);
        return v33;
      }
    }
  }
  else
  {
    v14 = *(_QWORD *)(a2 + 80);
    v15 = *(_QWORD *)(a2 + 48);
    v16 = *(_DWORD *)(a2 + 68);
    v34 = *a1;
    v36 = v14;
    if ( v14 )
    {
      v26 = v15;
      v27 = v16;
      v29 = v9;
      v31 = v10;
      sub_B96E90((__int64)&v36, v14, 1);
      LODWORD(v15) = v26;
      v16 = v27;
      v9 = v29;
      v10 = v31;
    }
    *((_QWORD *)&v24 + 1) = v10;
    *(_QWORD *)&v24 = v9;
    *((_QWORD *)&v22 + 1) = v5;
    *(_QWORD *)&v22 = v4;
    *((_QWORD *)&v21 + 1) = v8;
    *(_QWORD *)&v21 = v7;
    v37 = *(_DWORD *)(a2 + 72);
    result = sub_3412970(v34, 70, (unsigned int)&v36, v15, v16, (unsigned int)&v36, v21, v22, v24);
    if ( v36 )
    {
      v35 = result;
      sub_B91220((__int64)&v36, v36);
      return v35;
    }
  }
  return result;
}
