// Function: sub_32C2770
// Address: 0x32c2770
//
__int64 __fastcall sub_32C2770(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        int a8,
        char a9)
{
  __int64 v10; // rdx
  int v12; // eax
  __int64 v13; // r14
  __int64 v14; // r15
  int v15; // ecx
  __int128 *v16; // rsi
  __int64 v17; // rax
  int v18; // r9d
  int v19; // edx
  __int64 *v20; // rsi
  __int64 *v21; // rax
  char v22; // al
  __int64 v23; // rax
  __int64 v24; // rcx
  int v25; // [rsp-8h] [rbp-B8h]
  int v26; // [rsp+8h] [rbp-A8h]
  __int64 v27; // [rsp+20h] [rbp-90h] BYREF
  __int64 v28; // [rsp+28h] [rbp-88h]
  _QWORD v29[2]; // [rsp+30h] [rbp-80h] BYREF
  __int64 v30; // [rsp+40h] [rbp-70h] BYREF
  int v31; // [rsp+48h] [rbp-68h]
  char *v32; // [rsp+50h] [rbp-60h] BYREF
  __int64 v33; // [rsp+58h] [rbp-58h]
  __int64 *v34; // [rsp+60h] [rbp-50h]
  __int64 *v35; // [rsp+68h] [rbp-48h]
  __int64 v36; // [rsp+70h] [rbp-40h]
  _QWORD *v37; // [rsp+78h] [rbp-38h]

  v29[0] = a3;
  v29[1] = a4;
  v27 = a5;
  v28 = a6;
  if ( a8 != 1 || (_DWORD)v28 != 1 || *(_DWORD *)(a7 + 24) != 77 )
    return 0;
  v10 = v27;
  v31 = 0;
  v30 = 0;
  v12 = *(_DWORD *)(v27 + 24);
  if ( v12 == 72 )
  {
    v22 = sub_33CF170(*(_QWORD *)(*(_QWORD *)(v27 + 40) + 40LL), *(_QWORD *)(*(_QWORD *)(v27 + 40) + 48LL));
    v10 = v27;
    if ( v22 )
    {
      v23 = *(_QWORD *)(v27 + 40);
      v24 = *(_QWORD *)(v23 + 80);
      LODWORD(v23) = *(_DWORD *)(v23 + 88);
      v30 = v24;
      v31 = v23;
      goto LABEL_13;
    }
    v12 = *(_DWORD *)(v27 + 24);
  }
  if ( v12 != 77
    || !(unsigned __int8)sub_33CF4D0(
                           *(_QWORD *)(*(_QWORD *)(v10 + 40) + 40LL),
                           *(_QWORD *)(*(_QWORD *)(v10 + 40) + 48LL)) )
  {
    return 0;
  }
  v13 = *(_QWORD *)(*(_QWORD *)(v27 + 40) + 40LL);
  v14 = *(_QWORD *)(*(_QWORD *)(v27 + 48) + 24LL);
  v15 = *(unsigned __int16 *)(*(_QWORD *)(v27 + 48) + 16LL);
  v16 = *(__int128 **)(v13 + 80);
  v32 = (char *)v16;
  if ( v16 )
  {
    v26 = v15;
    sub_B96E90((__int64)&v32, (__int64)v16, 1);
    v15 = v26;
  }
  LODWORD(v33) = *(_DWORD *)(v13 + 72);
  v17 = sub_3400BD0(a2, 1, (unsigned int)&v32, v15, v14, 0, 0);
  v18 = v25;
  v30 = v17;
  v31 = v19;
  if ( v32 )
    sub_B91220((__int64)&v32, (__int64)v32);
  v10 = v27;
LABEL_13:
  v33 = a2;
  v32 = &a9;
  v34 = &v27;
  v35 = &v30;
  v37 = v29;
  v36 = a1;
  v20 = *(__int64 **)(v10 + 40);
  v21 = *(__int64 **)(a7 + 40);
  if ( *v20 == a7 && !*((_DWORD *)v20 + 2) )
    return sub_32C25A0((__int128 **)&v32, *v21, v21[1], v21[5], v21[6], v18);
  if ( *v21 != v10 || *((_DWORD *)v21 + 2) )
  {
    if ( v21[5] == v10 && !*((_DWORD *)v21 + 12) )
      return sub_32C25A0((__int128 **)&v32, *v21, v21[1], *v20, v20[1], *v21);
    return 0;
  }
  return sub_32C25A0((__int128 **)&v32, *v20, v20[1], v21[5], v21[6], *v20);
}
