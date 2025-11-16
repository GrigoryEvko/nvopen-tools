// Function: sub_12913D0
// Address: 0x12913d0
//
__int64 __fastcall sub_12913D0(_QWORD *a1, __int64 *a2, _QWORD *a3, __int64 a4, __int64 a5)
{
  char *i; // r14
  __int64 v7; // r15
  _QWORD *v8; // rax
  unsigned __int64 v9; // r9
  unsigned __int64 v10; // rdx
  _QWORD *v11; // rax
  __int64 v12; // rax
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // rax
  __int64 v17; // r15
  __int64 v18; // rbx
  __int64 v19; // rax
  __int64 v20; // rax
  char *v21; // rbx
  unsigned int v22; // r8d
  __int64 v23; // r15
  __int64 v24; // rax
  __int64 v25; // rax
  unsigned __int64 *v26; // r15
  __int64 v27; // rax
  unsigned __int64 v28; // rcx
  __int64 v29; // rsi
  __int64 v30; // rsi
  __int64 v31; // rax
  __int64 v32; // rax
  unsigned __int64 *v33; // r14
  __int64 v34; // rax
  unsigned __int64 v35; // rcx
  __int64 v36; // rsi
  _QWORD *v37; // rdx
  __int64 v38; // rsi
  unsigned int v39; // [rsp+8h] [rbp-B8h]
  __int64 v40; // [rsp+18h] [rbp-A8h] BYREF
  _QWORD v41[2]; // [rsp+20h] [rbp-A0h] BYREF
  __int16 v42; // [rsp+30h] [rbp-90h]
  _BYTE v43[16]; // [rsp+40h] [rbp-80h] BYREF
  __int16 v44; // [rsp+50h] [rbp-70h]
  char v45[8]; // [rsp+60h] [rbp-60h] BYREF
  char *v46; // [rsp+68h] [rbp-58h]

  if ( (_BYTE)a4 == 1 && !(_BYTE)a5 )
  {
    sub_1286D80((__int64)v45, a1, (__int64)a2, a4, a5);
    i = v46;
LABEL_4:
    v7 = a3[1];
    v8 = (_QWORD *)*a3;
    v9 = v7 + 1;
    if ( (_QWORD *)*a3 == a3 + 2 )
      v10 = 15;
    else
      v10 = a3[2];
    if ( v9 > v10 )
    {
      sub_2240BB0(a3, a3[1], 0, 0, 1);
      v8 = (_QWORD *)*a3;
      v9 = v7 + 1;
    }
    *((_BYTE *)v8 + v7) = 42;
    v11 = (_QWORD *)*a3;
    a3[1] = v9;
    *((_BYTE *)v11 + v7 + 1) = 0;
    goto LABEL_9;
  }
  if ( !sub_127B420(*a2) )
  {
    i = sub_128F980((__int64)a1, (__int64)a2);
    v12 = *(_QWORD *)i;
    if ( *(_BYTE *)(*(_QWORD *)i + 8LL) != 15 )
      return (__int64)i;
    goto LABEL_31;
  }
  sub_1286D80((__int64)v45, a1, (__int64)a2, v14, v15);
  v16 = *a2;
  for ( i = v46; *(_BYTE *)(v16 + 140) == 12; v16 = *(_QWORD *)(v16 + 160) )
    ;
  v17 = 8LL * *(_QWORD *)(v16 + 128);
  if ( (unsigned __int64)(v17 - 1) > 0x3F || ((v17 - 1) & v17) != 0 )
    goto LABEL_4;
  v18 = *(_QWORD *)v46;
  v19 = sub_1644900(a1[5], (unsigned int)v17);
  v20 = sub_1646BA0(v19, *(_DWORD *)(v18 + 8) >> 8);
  v42 = 257;
  if ( v20 == *(_QWORD *)i )
  {
    v21 = i;
  }
  else if ( (unsigned __int8)i[16] > 0x10u )
  {
    v44 = 257;
    v21 = (char *)sub_15FDBD0(47, i, v20, v43, 0);
    v32 = a1[7];
    if ( v32 )
    {
      v33 = (unsigned __int64 *)a1[8];
      sub_157E9D0(v32 + 40, v21);
      v34 = *((_QWORD *)v21 + 3);
      v35 = *v33;
      *((_QWORD *)v21 + 4) = v33;
      v35 &= 0xFFFFFFFFFFFFFFF8LL;
      *((_QWORD *)v21 + 3) = v35 | v34 & 7;
      *(_QWORD *)(v35 + 8) = v21 + 24;
      *v33 = *v33 & 7 | (unsigned __int64)(v21 + 24);
    }
    sub_164B780(v21, v41);
    v36 = a1[6];
    if ( v36 )
    {
      v40 = a1[6];
      sub_1623A60(&v40, v36, 2);
      v37 = v21 + 48;
      if ( *((_QWORD *)v21 + 6) )
      {
        sub_161E7C0(v21 + 48);
        v37 = v21 + 48;
      }
      v38 = v40;
      *((_QWORD *)v21 + 6) = v40;
      if ( v38 )
        sub_1623210(&v40, v38, v37);
    }
  }
  else
  {
    v21 = (char *)sub_15A46C0(47, i, v20, 0);
  }
  v44 = 257;
  v22 = unk_4D0463C;
  if ( unk_4D0463C )
    v22 = sub_126A420(a1[4], (unsigned __int64)v21);
  v39 = v22;
  v23 = sub_1644900(a1[5], (unsigned int)v17);
  v24 = sub_1648A60(64, 1);
  i = (char *)v24;
  if ( v24 )
    sub_15F9210(v24, v23, v21, 0, v39, 0);
  v25 = a1[7];
  if ( v25 )
  {
    v26 = (unsigned __int64 *)a1[8];
    sub_157E9D0(v25 + 40, i);
    v27 = *((_QWORD *)i + 3);
    v28 = *v26;
    *((_QWORD *)i + 4) = v26;
    v28 &= 0xFFFFFFFFFFFFFFF8LL;
    *((_QWORD *)i + 3) = v28 | v27 & 7;
    *(_QWORD *)(v28 + 8) = i + 24;
    *v26 = *v26 & 7 | (unsigned __int64)(i + 24);
  }
  sub_164B780(i, v43);
  v29 = a1[6];
  if ( v29 )
  {
    v41[0] = a1[6];
    sub_1623A60(v41, v29, 2);
    if ( *((_QWORD *)i + 6) )
      sub_161E7C0(i + 48);
    v30 = v41[0];
    *((_QWORD *)i + 6) = v41[0];
    if ( v30 )
      sub_1623210(v41, v30, i + 48);
  }
LABEL_9:
  v12 = *(_QWORD *)i;
  if ( *(_BYTE *)(*(_QWORD *)i + 8LL) != 15 )
    return (__int64)i;
LABEL_31:
  if ( !(*(_DWORD *)(v12 + 8) >> 8) )
    return (__int64)i;
  v31 = sub_1646BA0(*(_QWORD *)(v12 + 24), 0);
  return sub_1289630(a1, (__int64)i, v31);
}
