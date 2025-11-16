// Function: sub_233F660
// Address: 0x233f660
//
_QWORD *__fastcall sub_233F660(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  _QWORD *v4; // rax
  _QWORD *v5; // rbx
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v9; // [rsp+0h] [rbp-C0h] BYREF
  __int64 v10; // [rsp+8h] [rbp-B8h]
  __int64 v11; // [rsp+10h] [rbp-B0h]
  int v12; // [rsp+18h] [rbp-A8h]
  __int64 v13; // [rsp+20h] [rbp-A0h]
  __int64 v14; // [rsp+28h] [rbp-98h]
  __int64 v15; // [rsp+30h] [rbp-90h]
  __int64 v16; // [rsp+38h] [rbp-88h]
  __int64 v17; // [rsp+40h] [rbp-80h]
  __int64 v18; // [rsp+48h] [rbp-78h]
  __int64 v19; // [rsp+50h] [rbp-70h] BYREF
  __int64 v20; // [rsp+58h] [rbp-68h]
  __int64 v21; // [rsp+60h] [rbp-60h]
  int v22; // [rsp+68h] [rbp-58h]
  __int64 v23; // [rsp+70h] [rbp-50h]
  __int64 v24; // [rsp+78h] [rbp-48h]
  __int64 v25; // [rsp+80h] [rbp-40h]
  __int64 v26; // [rsp+88h] [rbp-38h]
  __int64 v27; // [rsp+90h] [rbp-30h]
  __int64 v28; // [rsp+98h] [rbp-28h]

  sub_D376B0((__int64)&v9, a2 + 8, a3, a4);
  ++v9;
  v19 = 1;
  v20 = v10;
  v10 = 0;
  v21 = v11;
  v11 = 0;
  v22 = v12;
  v12 = 0;
  v23 = v13;
  v24 = v14;
  v25 = v15;
  v26 = v16;
  v27 = v17;
  v28 = v18;
  v4 = (_QWORD *)sub_22077B0(0x58u);
  v5 = v4;
  if ( v4 )
  {
    ++v19;
    v4[1] = 1;
    *v4 = &unk_4A0B1C8;
    v6 = v20;
    v20 = 0;
    v5[2] = v6;
    v7 = v21;
    v21 = 0;
    v5[3] = v7;
    LODWORD(v7) = v22;
    v22 = 0;
    *((_DWORD *)v5 + 8) = v7;
    v5[5] = v23;
    v5[6] = v24;
    v5[7] = v25;
    v5[8] = v26;
    v5[9] = v27;
    v5[10] = v28;
  }
  sub_233F0E0((__int64)&v19);
  *a1 = v5;
  sub_233F0E0((__int64)&v9);
  return a1;
}
