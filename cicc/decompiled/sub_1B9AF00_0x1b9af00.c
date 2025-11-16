// Function: sub_1B9AF00
// Address: 0x1b9af00
//
__int64 *__fastcall sub_1B9AF00(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        unsigned __int64 a4,
        __int64 a5,
        double a6,
        double a7,
        double a8)
{
  char v10; // al
  __int64 v11; // rax
  __int64 v12; // rbx
  unsigned int v13; // r12d
  __int64 *result; // rax
  unsigned int v15; // ebx
  __int64 v16; // rsi
  __int64 v17; // r8
  __int64 v18; // rax
  __int64 v19; // rax
  int v20; // r8d
  unsigned __int64 *v21; // r9
  unsigned __int64 *v22; // r9
  __int64 v23; // rsi
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // r8
  char v27; // al
  __int64 v28; // rdx
  int v29; // r9d
  __int64 v30; // rdi
  __int64 v31; // rax
  __int64 v32; // rsi
  __int64 v33; // rsi
  __int64 v34; // rsi
  __int64 v35; // rdx
  unsigned __int8 *v36; // rsi
  int v37; // [rsp+10h] [rbp-C0h]
  __int64 *v38; // [rsp+10h] [rbp-C0h]
  __int64 v39; // [rsp+10h] [rbp-C0h]
  unsigned int v40; // [rsp+1Ch] [rbp-B4h]
  unsigned int *v41; // [rsp+20h] [rbp-B0h]
  __int64 v42; // [rsp+28h] [rbp-A8h]
  unsigned int v45; // [rsp+40h] [rbp-90h]
  int v46; // [rsp+44h] [rbp-8Ch]
  __int64 v48; // [rsp+50h] [rbp-80h]
  __int64 v49; // [rsp+50h] [rbp-80h]
  __int64 v50; // [rsp+50h] [rbp-80h]
  __int64 v51; // [rsp+50h] [rbp-80h]
  __int64 v52; // [rsp+50h] [rbp-80h]
  __int64 v53; // [rsp+50h] [rbp-80h]
  __int64 v54; // [rsp+50h] [rbp-80h]
  __int64 v55; // [rsp+58h] [rbp-78h]
  __int64 v56[2]; // [rsp+60h] [rbp-70h] BYREF
  __int16 v57; // [rsp+70h] [rbp-60h]
  __int64 v58[2]; // [rsp+80h] [rbp-50h] BYREF
  __int16 v59; // [rsp+90h] [rbp-40h]

  v55 = *(_QWORD *)a2;
  v10 = *(_BYTE *)(*(_QWORD *)a2 + 8LL);
  if ( v10 == 16 )
  {
    v55 = **(_QWORD **)(v55 + 16);
    v10 = *(_BYTE *)(v55 + 8);
  }
  if ( v10 == 11 )
  {
    v45 = 15;
    v40 = 11;
  }
  else
  {
    v11 = *(_QWORD *)(a5 + 40);
    v45 = 16;
    if ( v11 )
      v40 = *(unsigned __int8 *)(v11 + 16) - 24;
    else
      v40 = 29;
  }
  v12 = *(_QWORD *)(a1 + 456);
  v46 = *(_DWORD *)(a1 + 88);
  LODWORD(v56[0]) = v46;
  if ( v46 != 1 )
  {
    if ( (unsigned __int8)sub_1B97860(v12 + 168, (int *)v56, v58) )
    {
      if ( !sub_13A0E30(v58[0] + 8, a4) )
        goto LABEL_9;
    }
    else if ( !sub_13A0E30(*(_QWORD *)(v12 + 176) + 80LL * *(unsigned int *)(v12 + 192) + 8, a4) )
    {
LABEL_9:
      v46 = *(_DWORD *)(a1 + 88);
      goto LABEL_10;
    }
    v46 = 1;
  }
LABEL_10:
  v13 = 0;
  v42 = a1 + 96;
  v41 = (unsigned int *)(a1 + 280);
  result = v56;
  if ( *(_DWORD *)(a1 + 92) )
  {
    while ( 1 )
    {
      v15 = 0;
      if ( v46 )
        break;
LABEL_34:
      if ( *(_DWORD *)(a1 + 92) <= ++v13 )
        return result;
    }
    while ( 1 )
    {
      v23 = v13 * *(_DWORD *)(a1 + 88) + v15;
      if ( *(_BYTE *)(v55 + 8) == 11 )
      {
        v16 = sub_15A0930(v55, v23);
        v57 = 257;
        if ( *(_BYTE *)(v16 + 16) > 0x10u )
          goto LABEL_19;
      }
      else
      {
        a6 = (double)(int)v23;
        v16 = sub_15A10B0(v55, (double)(int)v23);
        v57 = 257;
        if ( *(_BYTE *)(v16 + 16) > 0x10u )
          goto LABEL_19;
      }
      if ( *(_BYTE *)(a3 + 16) > 0x10u || (v17 = sub_15A2A30((__int64 *)v45, (__int64 *)v16, a3, 0, 0, a6, a7, a8)) == 0 )
      {
LABEL_19:
        v59 = 257;
        v24 = sub_15FB440(v45, (__int64 *)v16, a3, (__int64)v58, 0);
        v25 = *(_QWORD *)v24;
        v26 = v24;
        v27 = *(_BYTE *)(*(_QWORD *)v24 + 8LL);
        if ( v27 == 16 )
          v27 = *(_BYTE *)(**(_QWORD **)(v25 + 16) + 8LL);
        if ( (unsigned __int8)(v27 - 1) <= 5u || *(_BYTE *)(v26 + 16) == 76 )
        {
          v28 = *(_QWORD *)(a1 + 128);
          v29 = *(_DWORD *)(a1 + 136);
          if ( v28 )
          {
            v37 = *(_DWORD *)(a1 + 136);
            v49 = v26;
            sub_1625C10(v26, 3, v28);
            v29 = v37;
            v26 = v49;
          }
          v50 = v26;
          sub_15F2440(v26, v29);
          v26 = v50;
        }
        v30 = *(_QWORD *)(a1 + 104);
        if ( v30 )
        {
          v51 = v26;
          v38 = *(__int64 **)(a1 + 112);
          sub_157E9D0(v30 + 40, v26);
          v26 = v51;
          v31 = *(_QWORD *)(v51 + 24);
          v32 = *v38;
          *(_QWORD *)(v51 + 32) = v38;
          v32 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v51 + 24) = v32 | v31 & 7;
          *(_QWORD *)(v32 + 8) = v51 + 24;
          *v38 = *v38 & 7 | (v51 + 24);
        }
        v52 = v26;
        sub_164B780(v26, v56);
        v33 = *(_QWORD *)(a1 + 96);
        v17 = v52;
        if ( v33 )
        {
          v58[0] = *(_QWORD *)(a1 + 96);
          sub_1623A60((__int64)v58, v33, 2);
          v17 = v52;
          v34 = *(_QWORD *)(v52 + 48);
          v35 = v52 + 48;
          if ( v34 )
          {
            v39 = v52;
            v53 = v52 + 48;
            sub_161E7C0(v53, v34);
            v17 = v39;
            v35 = v53;
          }
          v36 = (unsigned __int8 *)v58[0];
          *(_QWORD *)(v17 + 48) = v58[0];
          if ( v36 )
          {
            v54 = v17;
            sub_1623210((__int64)v58, v36, v35);
            v17 = v54;
          }
        }
      }
      v18 = sub_1B8ED40(v17);
      v59 = 257;
      v19 = sub_1904E90(v42, v40, a2, v18, v58, 0, a6, a7, a8);
      v48 = sub_1B8ED40(v19);
      v58[0] = __PAIR64__(v15, v13);
      sub_1B9A1B0(v41, a4, (unsigned int *)v58, v48, v20, v21);
      v22 = (unsigned __int64 *)v15++;
      result = (__int64 *)sub_1B9A880(a1, a5, a4, v48, (unsigned int *)v13, v22);
      if ( v46 == v15 )
        goto LABEL_34;
    }
  }
  return result;
}
