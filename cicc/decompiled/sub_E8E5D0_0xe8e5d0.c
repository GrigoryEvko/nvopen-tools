// Function: sub_E8E5D0
// Address: 0xe8e5d0
//
__int64 __fastcall sub_E8E5D0(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rbx
  __int64 v4; // r8
  __int64 v5; // rdx
  __int64 v6; // r12
  __int64 v7; // rax
  __int128 v8; // rax
  __int64 v10; // [rsp+0h] [rbp-160h]
  const char *v11; // [rsp+10h] [rbp-150h]
  __int64 v12; // [rsp+18h] [rbp-148h]
  char v13; // [rsp+30h] [rbp-130h]
  char v14; // [rsp+31h] [rbp-12Fh]
  __int128 v15; // [rsp+40h] [rbp-120h] BYREF
  const char *v16; // [rsp+50h] [rbp-110h]
  __int64 v17; // [rsp+58h] [rbp-108h]
  __int64 v18; // [rsp+60h] [rbp-100h]
  __int128 v19; // [rsp+70h] [rbp-F0h]
  __int16 v20; // [rsp+90h] [rbp-D0h]
  _QWORD v21[2]; // [rsp+A0h] [rbp-C0h] BYREF
  __int128 v22; // [rsp+B0h] [rbp-B0h]
  __int64 v23; // [rsp+C0h] [rbp-A0h]
  const char *v24; // [rsp+D0h] [rbp-90h]
  __int64 v25; // [rsp+D8h] [rbp-88h]
  __int64 v26; // [rsp+F0h] [rbp-70h]
  _QWORD v27[4]; // [rsp+100h] [rbp-60h] BYREF
  __int64 v28; // [rsp+120h] [rbp-40h]

  v4 = *(_QWORD *)(a1[36] + 8LL);
  if ( (*(_BYTE *)(v4 + 48) & 0x20) == 0 )
    return sub_E8E3C0(a1, a2, a3);
  v5 = *(_QWORD *)(v4 + 128);
  v6 = a1[1];
  v24 = "' cannot have instructions";
  v7 = *(_QWORD *)(v4 + 136);
  v20 = 261;
  *((_QWORD *)&v19 + 1) = v7;
  LOWORD(v26) = 259;
  *(_QWORD *)&v19 = v5;
  v14 = 1;
  v11 = " section '";
  v13 = 3;
  *(_QWORD *)&v8 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v4 + 16LL))(v4);
  v15 = v8;
  v16 = " section '";
  v17 = v12;
  LOWORD(v18) = 773;
  v21[0] = &v15;
  v21[1] = v3;
  v22 = v19;
  LOWORD(v23) = 1282;
  v27[0] = v21;
  v27[2] = "' cannot have instructions";
  v27[1] = v10;
  v27[3] = v25;
  LOWORD(v28) = 770;
  return sub_E66880(v6, *(_QWORD **)(a2 + 8), (__int64)v27);
}
