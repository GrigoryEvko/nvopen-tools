// Function: sub_3156B80
// Address: 0x3156b80
//
void __fastcall sub_3156B80(__int64 a1, _QWORD *a2)
{
  __int64 v3; // rsi
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // r12
  unsigned int v17; // ebx
  __int64 v18; // rsi
  __int64 v19; // rsi
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // r9
  _QWORD *v28; // [rsp+8h] [rbp-F8h]
  char v29; // [rsp+17h] [rbp-E9h] BYREF
  __int64 v30; // [rsp+18h] [rbp-E8h] BYREF
  __int64 v31; // [rsp+20h] [rbp-E0h] BYREF
  int v32; // [rsp+28h] [rbp-D8h] BYREF
  _QWORD *v33; // [rsp+30h] [rbp-D0h]
  int *v34; // [rsp+38h] [rbp-C8h]
  int *v35; // [rsp+40h] [rbp-C0h]
  __int64 v36; // [rsp+48h] [rbp-B8h]
  __m128i v37[11]; // [rsp+50h] [rbp-B0h] BYREF

  v3 = a1;
  sub_CB1A80((__int64)v37, a1, 0, 70);
  v29 = 0;
  v30 = 0;
  sub_CB05C0(v37, a1, v4, v5, v6, v7);
  if ( a2[5] )
  {
    sub_CB24B0((__int64)v37, "Contexts", 0, 0, &v29, &v30);
    sub_3154660((__int64)v37, (__int64)a2);
    v3 = 0;
    sub_CB0850((__int64)v37, 0, v8, v9, v10, v11);
    if ( !a2[11] )
      goto LABEL_3;
  }
  else if ( !a2[11] )
  {
    goto LABEL_3;
  }
  sub_CB24B0((__int64)v37, "FlatProfiles", 0, 0, &v29, &v30);
  sub_CB0550(v37, (__int64)"FlatProfiles", v12, v13, v14, v15);
  v16 = a2[9];
  v28 = a2 + 7;
  if ( (_QWORD *)v16 != a2 + 7 )
  {
    v17 = 0;
    do
    {
      v18 = v17++;
      sub_CB00A0((__int64)v37, v18, &v30);
      v35 = &v32;
      v36 = 0;
      v19 = *(_QWORD *)(v16 + 32);
      v32 = 0;
      v33 = 0;
      v34 = &v32;
      sub_3154400((__int64)v37, v19, v16 + 40, &v31);
      sub_31541A0(v33);
      sub_CB0910((__int64)v37, 0, v20, v21, v22, v23);
      v16 = sub_220EF30(v16);
    }
    while ( v28 != (_QWORD *)v16 );
  }
  sub_CB22A0(v37);
  v3 = 0;
  sub_CB0850((__int64)v37, 0, v24, v25, v26, v27);
LABEL_3:
  sub_CB2220(v37);
  sub_CB0A00(v37, v3);
}
