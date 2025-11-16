// Function: sub_302EAB0
// Address: 0x302eab0
//
__int64 __fastcall sub_302EAB0(int a1, int a2, __int64 a3, unsigned __int64 a4, __int64 a5, __int64 a6, char a7)
{
  unsigned int v9; // edx
  __int128 v10; // rax
  int v11; // r9d
  __int64 v12; // rax
  int v13; // r9d
  __int64 v14; // r10
  __int64 v15; // r11
  unsigned int v16; // edx
  __int128 v17; // rax
  int v18; // r9d
  unsigned int v20; // edx
  __int128 v21; // [rsp-20h] [rbp-B0h]
  __int128 v22; // [rsp-10h] [rbp-A0h]
  __int128 v23; // [rsp+0h] [rbp-90h]
  __int64 v24; // [rsp+18h] [rbp-78h]
  __int128 v25; // [rsp+20h] [rbp-70h]
  __int64 v26; // [rsp+50h] [rbp-40h]

  *(_QWORD *)&v25 = a5;
  *((_QWORD *)&v25 + 1) = a6;
  if ( a7 )
  {
    v26 = sub_3400BD0(a1, 0xFFFF, a2, 7, 0, 0, 0);
    v24 = v9;
    *(_QWORD *)&v10 = sub_3400BD0(a1, 16, a2, 7, 0, 0, 0);
    *((_QWORD *)&v21 + 1) = a4;
    *(_QWORD *)&v21 = a3;
    v12 = sub_3406EB0(a1, 190, a2, 7, 0, v11, v21, v10);
    v14 = v26;
    v15 = v24;
    a3 = v12;
    a4 = v16 | a4 & 0xFFFFFFFF00000000LL;
  }
  else
  {
    v14 = sub_3400BD0(a1, -65536, a2, 7, 0, 0, 0);
    v15 = v20;
  }
  *((_QWORD *)&v23 + 1) = v15;
  *(_QWORD *)&v23 = v14;
  *(_QWORD *)&v17 = sub_3406EB0(a1, 186, a2, 7, 0, v13, v25, v23);
  *((_QWORD *)&v22 + 1) = a4;
  *(_QWORD *)&v22 = a3;
  return sub_3406EB0(a1, 187, a2, 7, 0, v18, v22, v17);
}
