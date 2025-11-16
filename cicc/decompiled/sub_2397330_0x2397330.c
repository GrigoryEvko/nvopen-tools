// Function: sub_2397330
// Address: 0x2397330
//
_QWORD *__fastcall sub_2397330(_QWORD *a1, __int64 a2, __int64 a3)
{
  _QWORD *v3; // rax
  _QWORD *v4; // rbx
  int v5; // eax
  __int64 v6; // rax
  __int64 v7; // rax
  int v9; // [rsp+0h] [rbp-140h] BYREF
  __int64 v10; // [rsp+8h] [rbp-138h]
  __int64 v11; // [rsp+10h] [rbp-130h]
  __int64 v12; // [rsp+18h] [rbp-128h]
  int v13; // [rsp+20h] [rbp-120h]
  __int64 v14; // [rsp+28h] [rbp-118h]
  __int64 v15; // [rsp+30h] [rbp-110h]
  __int64 v16; // [rsp+38h] [rbp-108h]
  int v17; // [rsp+40h] [rbp-100h]
  __int64 v18; // [rsp+48h] [rbp-F8h]
  __int64 v19; // [rsp+50h] [rbp-F0h]
  __int64 v20; // [rsp+58h] [rbp-E8h]
  int v21; // [rsp+60h] [rbp-E0h]
  __int64 v22; // [rsp+68h] [rbp-D8h]
  __int64 v23; // [rsp+70h] [rbp-D0h]
  __int64 v24; // [rsp+78h] [rbp-C8h]
  int v25; // [rsp+80h] [rbp-C0h]
  __int64 v26; // [rsp+88h] [rbp-B8h]
  int v27; // [rsp+90h] [rbp-B0h] BYREF
  __int64 v28; // [rsp+98h] [rbp-A8h]
  __int64 v29; // [rsp+A0h] [rbp-A0h]
  __int64 v30; // [rsp+A8h] [rbp-98h]
  int v31; // [rsp+B0h] [rbp-90h]
  __int64 v32; // [rsp+B8h] [rbp-88h]
  __int64 v33; // [rsp+C0h] [rbp-80h]
  __int64 v34; // [rsp+C8h] [rbp-78h]
  int v35; // [rsp+D0h] [rbp-70h]
  __int64 v36; // [rsp+D8h] [rbp-68h]
  __int64 v37; // [rsp+E0h] [rbp-60h]
  __int64 v38; // [rsp+E8h] [rbp-58h]
  int v39; // [rsp+F0h] [rbp-50h]
  __int64 v40; // [rsp+F8h] [rbp-48h]
  __int64 v41; // [rsp+100h] [rbp-40h]
  __int64 v42; // [rsp+108h] [rbp-38h]
  int v43; // [rsp+110h] [rbp-30h]
  __int64 v44; // [rsp+118h] [rbp-28h]

  sub_22D5C30((__int64)&v9, a2 + 8, a3);
  ++v10;
  ++v14;
  v27 = v9;
  ++v18;
  v29 = v11;
  v28 = 1;
  v30 = v12;
  v11 = 0;
  v31 = v13;
  v12 = 0;
  v33 = v15;
  v13 = 0;
  v34 = v16;
  v32 = 1;
  v35 = v17;
  v15 = 0;
  v37 = v19;
  v16 = 0;
  v17 = 0;
  v36 = 1;
  v38 = v20;
  ++v22;
  v39 = v21;
  v19 = 0;
  v41 = v23;
  v20 = 0;
  v42 = v24;
  v21 = 0;
  v43 = v25;
  v40 = 1;
  v23 = 0;
  v24 = 0;
  v25 = 0;
  v44 = v26;
  v3 = (_QWORD *)sub_22077B0(0x98u);
  v4 = v3;
  if ( v3 )
  {
    ++v28;
    ++v32;
    ++v36;
    *v3 = &unk_4A0AE80;
    v5 = v27;
    v4[2] = 1;
    *((_DWORD *)v4 + 2) = v5;
    v6 = v29;
    v4[6] = 1;
    v4[3] = v6;
    v29 = 0;
    v4[4] = v30;
    v30 = 0;
    *((_DWORD *)v4 + 10) = v31;
    v31 = 0;
    v4[7] = v33;
    v33 = 0;
    v4[8] = v34;
    v34 = 0;
    *((_DWORD *)v4 + 18) = v35;
    v7 = v37;
    v35 = 0;
    v4[10] = 1;
    v4[11] = v7;
    ++v40;
    v4[12] = v38;
    v37 = 0;
    *((_DWORD *)v4 + 26) = v39;
    v38 = 0;
    v4[15] = v41;
    v39 = 0;
    v4[16] = v42;
    LODWORD(v7) = v43;
    v4[14] = 1;
    *((_DWORD *)v4 + 34) = v7;
    v41 = 0;
    v42 = 0;
    v43 = 0;
    v4[18] = v44;
  }
  sub_2397090((__int64)&v27);
  *a1 = v4;
  sub_2397090((__int64)&v9);
  return a1;
}
