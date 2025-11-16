// Function: sub_1280460
// Address: 0x1280460
//
__int64 __fastcall sub_1280460(
        __int64 a1,
        _QWORD *a2,
        _BYTE *a3,
        __int64 a4,
        unsigned int a5,
        __int64 a6,
        char a7,
        unsigned __int8 a8)
{
  __int64 v11; // r12
  __int64 v12; // rdi
  unsigned int v14; // eax
  unsigned int v15; // r15d
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rdi
  __int64 v19; // rax
  __int64 v20; // rax
  bool v21; // cc
  __int64 v22; // r15
  int v23; // ecx
  unsigned __int64 v24; // rbx
  unsigned __int64 v25; // rax
  unsigned __int64 v26; // rdx
  unsigned int v27; // r14d
  __int64 v28; // rax
  __int64 v29; // rdx
  char v30; // al
  __int64 v31; // rax
  __int64 *v32; // r14
  __int64 v33; // rax
  __int64 v34; // rcx
  __int64 v35; // rsi
  __int64 v36; // rsi
  __int64 v37; // rax
  __int64 v38; // rcx
  __int64 v39; // rax
  __int64 v40; // rax
  __int64 v41; // rax
  __int64 v42; // rcx
  __int64 v43; // r10
  __int64 v44; // rax
  __int64 v45; // rdi
  __int64 v46; // rcx
  __int64 v47; // rax
  __int64 v48; // rsi
  __int64 v49; // rdx
  __int64 v50; // rsi
  __int64 v51; // rax
  __int64 v52; // rax
  __int64 v53; // [rsp+8h] [rbp-C8h]
  unsigned int v54; // [rsp+10h] [rbp-C0h]
  __int64 v55; // [rsp+18h] [rbp-B8h]
  unsigned __int64 v56; // [rsp+28h] [rbp-A8h]
  __int64 v57; // [rsp+28h] [rbp-A8h]
  __int64 v58; // [rsp+28h] [rbp-A8h]
  __int64 *v59; // [rsp+28h] [rbp-A8h]
  __int64 v62; // [rsp+48h] [rbp-88h] BYREF
  __int64 v63; // [rsp+50h] [rbp-80h] BYREF
  __int64 v64; // [rsp+58h] [rbp-78h]
  _QWORD v65[2]; // [rsp+60h] [rbp-70h] BYREF
  char v66; // [rsp+70h] [rbp-60h]
  char v67; // [rsp+71h] [rbp-5Fh]
  _BYTE v68[16]; // [rsp+80h] [rbp-50h] BYREF
  __int16 v69; // [rsp+90h] [rbp-40h]

  if ( *(_BYTE *)(a6 + 24) != 4 )
    sub_127B550("unexpected field expression kind!", (_DWORD *)(a6 + 36), 1);
  v11 = *(_QWORD *)(a6 + 56);
  v12 = *(_QWORD *)(v11 + 120);
  if ( (*(_BYTE *)(v12 + 140) & 0xFB) == 8 )
  {
    v58 = a4;
    v30 = sub_8D4C10(v12, dword_4F077C4 != 2);
    a4 = v58;
    if ( (v30 & 2) != 0 )
      a8 = 1;
  }
  if ( (*(_BYTE *)(v11 + 144) & 4) != 0 )
  {
    sub_1280430(a1, (__int64)a2, (__int64)a3, a4, v11, a5, a8);
  }
  else
  {
    v56 = a4;
    v14 = sub_1277B60(a2[4] + 8LL, v11);
    v67 = 1;
    v15 = v14;
    v66 = 3;
    v65[0] = "tmp";
    v57 = sub_127A030(a2[4] + 8LL, v56, 0);
    v16 = sub_1643350(a2[9]);
    v17 = sub_159C470(v16, 0, 0);
    v18 = a2[9];
    v63 = v17;
    v19 = sub_1643350(v18);
    v20 = sub_159C470(v19, v15, 0);
    v21 = a3[16] <= 0x10u;
    v64 = v20;
    if ( v21 )
    {
      v68[4] = 0;
      v22 = sub_15A2E80(v57, (_DWORD)a3, (unsigned int)&v63, 2, 1, (unsigned int)v68, 0);
    }
    else
    {
      v69 = 257;
      if ( !v57 )
      {
        v52 = *(_QWORD *)a3;
        if ( *(_BYTE *)(*(_QWORD *)a3 + 8LL) == 16 )
          v52 = **(_QWORD **)(v52 + 16);
        v57 = *(_QWORD *)(v52 + 24);
      }
      v37 = sub_1648A60(72, 3);
      v22 = v37;
      if ( v37 )
      {
        v55 = v37;
        v38 = v37 - 72;
        v39 = *(_QWORD *)a3;
        if ( *(_BYTE *)(*(_QWORD *)a3 + 8LL) == 16 )
          v39 = **(_QWORD **)(v39 + 16);
        v53 = v38;
        v54 = *(_DWORD *)(v39 + 8) >> 8;
        v40 = sub_15F9F50(v57, &v63, 2);
        v41 = sub_1646BA0(v40, v54);
        v42 = v53;
        v43 = v41;
        v44 = *(_QWORD *)a3;
        if ( *(_BYTE *)(*(_QWORD *)a3 + 8LL) == 16
          || (v44 = *(_QWORD *)v63, *(_BYTE *)(*(_QWORD *)v63 + 8LL) == 16)
          || (v44 = *(_QWORD *)v64, *(_BYTE *)(*(_QWORD *)v64 + 8LL) == 16) )
        {
          v51 = sub_16463B0(v43, *(_QWORD *)(v44 + 32));
          v42 = v53;
          v43 = v51;
        }
        sub_15F1EA0(v22, v43, 32, v42, 3, 0);
        *(_QWORD *)(v22 + 56) = v57;
        *(_QWORD *)(v22 + 64) = sub_15F9F50(v57, &v63, 2);
        sub_15F9CE0(v22, a3, &v63, 2, v68);
      }
      else
      {
        v55 = 0;
      }
      sub_15FA2E0(v22, 1);
      v45 = a2[7];
      if ( v45 )
      {
        v59 = (__int64 *)a2[8];
        sub_157E9D0(v45 + 40, v22);
        v46 = *v59;
        v47 = *(_QWORD *)(v22 + 24) & 7LL;
        *(_QWORD *)(v22 + 32) = v59;
        v46 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v22 + 24) = v46 | v47;
        *(_QWORD *)(v46 + 8) = v22 + 24;
        *v59 = *v59 & 7 | (v22 + 24);
      }
      sub_164B780(v55, v65);
      v48 = a2[6];
      if ( v48 )
      {
        v62 = a2[6];
        sub_1623A60(&v62, v48, 2);
        v49 = v22 + 48;
        if ( *(_QWORD *)(v22 + 48) )
        {
          sub_161E7C0(v22 + 48);
          v49 = v22 + 48;
        }
        v50 = v62;
        *(_QWORD *)(v22 + 48) = v62;
        if ( v50 )
          sub_1623210(&v62, v50, v49);
      }
    }
    if ( (*(_BYTE *)(v11 + 145) & 0x10) != 0 || a7 )
    {
      v27 = *(_DWORD *)(*(_QWORD *)a3 + 8LL) >> 8;
      v28 = sub_127A040(a2[4] + 8LL, *(_QWORD *)(v11 + 120));
      v67 = 1;
      v66 = 3;
      v65[0] = "tmp";
      v29 = sub_1646BA0(v28, v27);
      if ( v29 != *(_QWORD *)v22 )
      {
        if ( *(_BYTE *)(v22 + 16) > 0x10u )
        {
          v69 = 257;
          v22 = sub_15FDBD0(47, v22, v29, v68, 0);
          v31 = a2[7];
          if ( v31 )
          {
            v32 = (__int64 *)a2[8];
            sub_157E9D0(v31 + 40, v22);
            v33 = *(_QWORD *)(v22 + 24);
            v34 = *v32;
            *(_QWORD *)(v22 + 32) = v32;
            v34 &= 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)(v22 + 24) = v34 | v33 & 7;
            *(_QWORD *)(v34 + 8) = v22 + 24;
            *v32 = *v32 & 7 | (v22 + 24);
          }
          sub_164B780(v22, v65);
          v35 = a2[6];
          if ( v35 )
          {
            v63 = a2[6];
            sub_1623A60(&v63, v35, 2);
            if ( *(_QWORD *)(v22 + 48) )
              sub_161E7C0(v22 + 48);
            v36 = v63;
            *(_QWORD *)(v22 + 48) = v63;
            if ( v36 )
              sub_1623210(&v63, v36, v22 + 48);
          }
        }
        else
        {
          v22 = sub_15A46C0(47, v22, v29, 0);
        }
      }
    }
    v23 = a8;
    v24 = a5;
    v25 = *(_QWORD *)(v11 + 128);
    if ( a5 )
    {
      while ( 1 )
      {
        v26 = v25 % v24;
        v25 = v24;
        if ( !v26 )
          break;
        v24 = v26;
      }
    }
    else
    {
      v24 = *(_QWORD *)(v11 + 128);
    }
    *(_DWORD *)a1 = 0;
    *(_QWORD *)(a1 + 8) = v22;
    *(_DWORD *)(a1 + 40) = v23;
    *(_DWORD *)(a1 + 16) = v24;
  }
  return a1;
}
