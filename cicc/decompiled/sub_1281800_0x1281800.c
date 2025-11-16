// Function: sub_1281800
// Address: 0x1281800
//
_QWORD *__fastcall sub_1281800(
        __int64 *a1,
        _DWORD *a2,
        unsigned __int64 *a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        int a7,
        _BYTE *a8,
        int a9,
        __int64 a10,
        int a11,
        char a12)
{
  __int64 v13; // r14
  __int64 v14; // r13
  unsigned int v15; // esi
  __int64 v16; // rdx
  __int64 i; // rax
  __int64 v18; // rax
  __int64 v19; // rax
  unsigned __int64 v20; // rdx
  int v21; // r8d
  __int64 v22; // rax
  _QWORD *v23; // r14
  __int64 v24; // rdi
  unsigned __int64 *v25; // r13
  __int64 v26; // rax
  unsigned __int64 v27; // rcx
  __int64 v28; // rsi
  __int64 v29; // rsi
  int v30; // eax
  int v31; // edx
  int v32; // eax
  bool v34; // al
  __int64 v35; // rax
  __int64 v36; // rsi
  __int64 v37; // rax
  __int64 v38; // rsi
  __int64 v39; // rdx
  __int64 v40; // rsi
  unsigned int v43; // [rsp+10h] [rbp-A0h]
  unsigned __int64 v44; // [rsp+18h] [rbp-98h]
  unsigned __int64 v45; // [rsp+20h] [rbp-90h]
  unsigned __int64 v46; // [rsp+20h] [rbp-90h]
  unsigned __int64 v47; // [rsp+28h] [rbp-88h]
  __int64 *v48; // [rsp+28h] [rbp-88h]
  __int64 v49; // [rsp+38h] [rbp-78h] BYREF
  _QWORD v50[2]; // [rsp+40h] [rbp-70h] BYREF
  char v51; // [rsp+50h] [rbp-60h]
  char v52; // [rsp+51h] [rbp-5Fh]
  _QWORD v53[2]; // [rsp+60h] [rbp-50h] BYREF
  __int16 v54; // [rsp+70h] [rbp-40h]

  if ( a7 != 1 )
    sub_127B550("error generating code for loading from bitfield!", a2, 1);
  v13 = (__int64)a8;
  v14 = sub_127A040(a1[4] + 8, *(_QWORD *)(a10 + 120));
  v15 = *(_DWORD *)(*(_QWORD *)a8 + 8LL);
  v52 = 1;
  v50[0] = "tmp";
  v51 = 3;
  v16 = sub_1646BA0(v14, v15 >> 8);
  if ( v16 != *(_QWORD *)a8 )
  {
    if ( a8[16] > 0x10u )
    {
      v54 = 257;
      v13 = sub_15FDBD0(47, a8, v16, v53, 0);
      v35 = a1[7];
      if ( v35 )
      {
        v48 = (__int64 *)a1[8];
        sub_157E9D0(v35 + 40, v13);
        v36 = *v48;
        v37 = *(_QWORD *)(v13 + 24) & 7LL;
        *(_QWORD *)(v13 + 32) = v48;
        v36 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v13 + 24) = v36 | v37;
        *(_QWORD *)(v36 + 8) = v13 + 24;
        *v48 = *v48 & 7 | (v13 + 24);
      }
      sub_164B780(v13, v50);
      v38 = a1[6];
      if ( v38 )
      {
        v49 = a1[6];
        sub_1623A60(&v49, v38, 2);
        v39 = v13 + 48;
        if ( *(_QWORD *)(v13 + 48) )
        {
          sub_161E7C0(v13 + 48);
          v39 = v13 + 48;
        }
        v40 = v49;
        *(_QWORD *)(v13 + 48) = v49;
        if ( v40 )
          sub_1623210(&v49, v40, v39);
      }
    }
    else
    {
      v13 = sub_15A46C0(47, a8, v16, 0);
    }
  }
  for ( i = *(_QWORD *)(a10 + 120); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v44 = *(_QWORD *)(i + 128);
  v47 = *(_QWORD *)(a10 + 128) / v44;
  v18 = sub_1643350(a1[5]);
  v19 = sub_159C470(v18, v47, 0);
  v53[0] = "tmp";
  v54 = 259;
  v20 = sub_12815B0(a1 + 6, v14, (_BYTE *)v13, v19, (__int64)v53);
  if ( a3 )
    *a3 = v20;
  if ( unk_4D0463C && (v46 = v20, v34 = sub_126A420(a1[4], v20), v20 = v46, v34) )
    v21 = 1;
  else
    v21 = a12 & 1;
  v43 = v21;
  v45 = v20;
  v53[0] = "tmp";
  v54 = 259;
  v22 = sub_1648A60(64, 1);
  v23 = (_QWORD *)v22;
  if ( v22 )
    sub_15F9210(v22, v14, v45, 0, v43, 0);
  v24 = a1[7];
  if ( v24 )
  {
    v25 = (unsigned __int64 *)a1[8];
    sub_157E9D0(v24 + 40, v23);
    v26 = v23[3];
    v27 = *v25;
    v23[4] = v25;
    v27 &= 0xFFFFFFFFFFFFFFF8LL;
    v23[3] = v27 | v26 & 7;
    *(_QWORD *)(v27 + 8) = v23 + 3;
    *v25 = *v25 & 7 | (unsigned __int64)(v23 + 3);
  }
  sub_164B780(v23, v53);
  v28 = a1[6];
  if ( v28 )
  {
    v50[0] = a1[6];
    sub_1623A60(v50, v28, 2);
    if ( v23[6] )
      sub_161E7C0(v23 + 6);
    v29 = v50[0];
    v23[6] = v50[0];
    if ( v29 )
      sub_1623210(v50, v29, v23 + 6);
  }
  v30 = *(unsigned __int8 *)(a10 + 137) + *(unsigned __int8 *)(a10 + 136);
  v31 = v30 + 6;
  v32 = v30 - 1;
  if ( v32 < 0 )
    v32 = v31;
  if ( v47 != (*(_QWORD *)(a10 + 128) + (v32 >> 3)) / v44 )
    sub_127B550("a bitfield straddling elements of container type is not supported!", a2, 1);
  return v23;
}
