// Function: sub_1286300
// Address: 0x1286300
//
__int64 __fastcall sub_1286300(__int64 *a1, __int64 a2, _BYTE *a3, unsigned int a4, unsigned int a5, __int64 a6)
{
  __int64 v9; // r12
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rdi
  __int64 v14; // rax
  __int64 v15; // rax
  bool v16; // cc
  _QWORD *v17; // r15
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // r10
  __int64 v23; // rax
  __int64 v24; // rdi
  unsigned __int64 *v25; // r12
  __int64 v26; // rax
  unsigned __int64 v27; // rcx
  __int64 v28; // rsi
  __int64 v29; // rsi
  __int64 v30; // rax
  unsigned int v31; // [rsp+4h] [rbp-8Ch]
  __int64 v32; // [rsp+8h] [rbp-88h]
  __int64 v33; // [rsp+10h] [rbp-80h]
  __int64 v35; // [rsp+28h] [rbp-68h] BYREF
  __int64 v36; // [rsp+30h] [rbp-60h] BYREF
  __int64 v37; // [rsp+38h] [rbp-58h]
  _BYTE v38[16]; // [rsp+40h] [rbp-50h] BYREF
  __int16 v39; // [rsp+50h] [rbp-40h]

  v9 = a2;
  v11 = sub_1643350(a1[3]);
  v12 = sub_159C470(v11, a4, 0);
  v13 = a1[3];
  v36 = v12;
  v14 = sub_1643350(v13);
  v15 = sub_159C470(v14, a5, 0);
  v16 = a3[16] <= 0x10u;
  v37 = v15;
  if ( v16 )
  {
    v38[4] = 0;
    return sub_15A2E80(a2, (_DWORD)a3, (unsigned int)&v36, 2, 1, (unsigned int)v38, 0);
  }
  else
  {
    v39 = 257;
    if ( !a2 )
    {
      v30 = *(_QWORD *)a3;
      if ( *(_BYTE *)(*(_QWORD *)a3 + 8LL) == 16 )
        v30 = **(_QWORD **)(v30 + 16);
      v9 = *(_QWORD *)(v30 + 24);
    }
    v19 = sub_1648A60(72, 3);
    v17 = (_QWORD *)v19;
    if ( v19 )
    {
      v33 = v19;
      v32 = v19 - 72;
      v20 = *(_QWORD *)a3;
      if ( *(_BYTE *)(*(_QWORD *)a3 + 8LL) == 16 )
        v20 = **(_QWORD **)(v20 + 16);
      v31 = *(_DWORD *)(v20 + 8) >> 8;
      v21 = sub_15F9F50(v9, &v36, 2);
      v22 = sub_1646BA0(v21, v31);
      v23 = *(_QWORD *)a3;
      if ( *(_BYTE *)(*(_QWORD *)a3 + 8LL) == 16
        || (v23 = *(_QWORD *)v36, *(_BYTE *)(*(_QWORD *)v36 + 8LL) == 16)
        || (v23 = *(_QWORD *)v37, *(_BYTE *)(*(_QWORD *)v37 + 8LL) == 16) )
      {
        v22 = sub_16463B0(v22, *(_QWORD *)(v23 + 32));
      }
      sub_15F1EA0(v17, v22, 32, v32, 3, 0);
      v17[7] = v9;
      v17[8] = sub_15F9F50(v9, &v36, 2);
      sub_15F9CE0(v17, a3, &v36, 2, v38);
    }
    else
    {
      v33 = 0;
    }
    sub_15FA2E0(v17, 1);
    v24 = a1[1];
    if ( v24 )
    {
      v25 = (unsigned __int64 *)a1[2];
      sub_157E9D0(v24 + 40, v17);
      v26 = v17[3];
      v27 = *v25;
      v17[4] = v25;
      v27 &= 0xFFFFFFFFFFFFFFF8LL;
      v17[3] = v27 | v26 & 7;
      *(_QWORD *)(v27 + 8) = v17 + 3;
      *v25 = *v25 & 7 | (unsigned __int64)(v17 + 3);
    }
    sub_164B780(v33, a6);
    v28 = *a1;
    if ( *a1 )
    {
      v35 = *a1;
      sub_1623A60(&v35, v28, 2);
      if ( v17[6] )
        sub_161E7C0(v17 + 6);
      v29 = v35;
      v17[6] = v35;
      if ( v29 )
        sub_1623210(&v35, v29, v17 + 6);
    }
  }
  return (__int64)v17;
}
