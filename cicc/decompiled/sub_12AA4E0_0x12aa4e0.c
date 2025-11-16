// Function: sub_12AA4E0
// Address: 0x12aa4e0
//
__int64 __fastcall sub_12AA4E0(__int64 a1, _QWORD *a2, int a3, __int64 a4, char a5)
{
  int v6; // r14d
  char *v8; // r15
  unsigned int v9; // r14d
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // r14
  __int64 v13; // rax
  __int64 v15; // rax
  __int64 v16; // r15
  __int64 v17; // rdi
  __int64 v18; // rsi
  __int64 v19; // rax
  __int64 v20; // rsi
  __int64 v21; // rdx
  __int64 v22; // rsi
  unsigned int v23; // eax
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rdi
  unsigned __int64 *v28; // r14
  __int64 v29; // rax
  unsigned __int64 v30; // rcx
  __int64 v31; // rsi
  char *v32; // rdx
  __int64 v33; // rsi
  __int64 v34; // rax
  __int64 v35; // rdi
  __int64 *v36; // r14
  __int64 v37; // rax
  __int64 v38; // rcx
  __int64 v39; // rsi
  __int64 v40; // rdx
  __int64 v41; // rsi
  _QWORD *v42; // [rsp+8h] [rbp-98h]
  __int64 v43; // [rsp+10h] [rbp-90h]
  __int64 *v44; // [rsp+10h] [rbp-90h]
  __int64 v46; // [rsp+28h] [rbp-78h] BYREF
  _QWORD v47[2]; // [rsp+30h] [rbp-70h] BYREF
  __int16 v48; // [rsp+40h] [rbp-60h]
  _QWORD v49[2]; // [rsp+50h] [rbp-50h] BYREF
  __int16 v50; // [rsp+60h] [rbp-40h]

  v6 = a3 - 194;
  v8 = sub_128F980((__int64)a2, *(_QWORD *)(*(_QWORD *)(a4 + 72) + 16LL));
  switch ( v6 )
  {
    case 0:
    case 4:
      v9 = 3;
      break;
    case 1:
    case 5:
      v9 = 4;
      break;
    case 2:
    case 6:
      v9 = 5;
      break;
    default:
      v9 = 1;
      break;
  }
  v10 = sub_1643330(a2[5]);
  v11 = sub_1646BA0(v10, v9);
  v12 = v11;
  if ( a5 )
  {
    v48 = 257;
    if ( v11 != *(_QWORD *)v8 )
    {
      if ( (unsigned __int8)v8[16] > 0x10u )
      {
        v50 = 257;
        v26 = sub_15FDBD0(46, v8, v11, v49, 0);
        v27 = a2[7];
        v8 = (char *)v26;
        if ( v27 )
        {
          v28 = (unsigned __int64 *)a2[8];
          sub_157E9D0(v27 + 40, v26);
          v29 = *((_QWORD *)v8 + 3);
          v30 = *v28;
          *((_QWORD *)v8 + 4) = v28;
          v30 &= 0xFFFFFFFFFFFFFFF8LL;
          *((_QWORD *)v8 + 3) = v30 | v29 & 7;
          *(_QWORD *)(v30 + 8) = v8 + 24;
          *v28 = *v28 & 7 | (unsigned __int64)(v8 + 24);
        }
        sub_164B780(v8, v47);
        v31 = a2[6];
        if ( v31 )
        {
          v46 = a2[6];
          sub_1623A60(&v46, v31, 2);
          v32 = v8 + 48;
          if ( *((_QWORD *)v8 + 6) )
          {
            sub_161E7C0(v8 + 48);
            v32 = v8 + 48;
          }
          v33 = v46;
          *((_QWORD *)v8 + 6) = v46;
          if ( v33 )
            sub_1623210(&v46, v33, v32);
        }
      }
      else
      {
        v8 = (char *)sub_15A46C0(46, v8, v11, 0);
      }
    }
    v13 = sub_1289750(a2, (__int64)v8);
    *(_BYTE *)(a1 + 12) &= ~1u;
    *(_QWORD *)a1 = v13;
    *(_DWORD *)(a1 + 8) = 0;
    *(_DWORD *)(a1 + 16) = 0;
  }
  else
  {
    v49[0] = "temp";
    v50 = 259;
    v42 = sub_127FC40(a2, *(_QWORD *)v8, (__int64)v49, 0, 0);
    sub_12A8F50(a2 + 6, (__int64)v8, (__int64)v42, 0);
    v50 = 257;
    v43 = *(_QWORD *)v8;
    v15 = sub_1648A60(64, 1);
    v16 = v15;
    if ( v15 )
      sub_15F9210(v15, v43, v42, 0, 0, 0);
    v17 = a2[7];
    if ( v17 )
    {
      v44 = (__int64 *)a2[8];
      sub_157E9D0(v17 + 40, v16);
      v18 = *v44;
      v19 = *(_QWORD *)(v16 + 24) & 7LL;
      *(_QWORD *)(v16 + 32) = v44;
      v18 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v16 + 24) = v18 | v19;
      *(_QWORD *)(v18 + 8) = v16 + 24;
      *v44 = *v44 & 7 | (v16 + 24);
    }
    sub_164B780(v16, v49);
    v20 = a2[6];
    if ( v20 )
    {
      v47[0] = a2[6];
      sub_1623A60(v47, v20, 2);
      v21 = v16 + 48;
      if ( *(_QWORD *)(v16 + 48) )
      {
        sub_161E7C0(v16 + 48);
        v21 = v16 + 48;
      }
      v22 = v47[0];
      *(_QWORD *)(v16 + 48) = v47[0];
      if ( v22 )
        sub_1623210(v47, v22, v21);
    }
    v48 = 257;
    if ( v12 != *(_QWORD *)v16 )
    {
      if ( *(_BYTE *)(v16 + 16) > 0x10u )
      {
        v50 = 257;
        v34 = sub_15FDFF0(v16, v12, v49, 0);
        v35 = a2[7];
        v16 = v34;
        if ( v35 )
        {
          v36 = (__int64 *)a2[8];
          sub_157E9D0(v35 + 40, v34);
          v37 = *(_QWORD *)(v16 + 24);
          v38 = *v36;
          *(_QWORD *)(v16 + 32) = v36;
          v38 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v16 + 24) = v38 | v37 & 7;
          *(_QWORD *)(v38 + 8) = v16 + 24;
          *v36 = *v36 & 7 | (v16 + 24);
        }
        sub_164B780(v16, v47);
        v39 = a2[6];
        if ( v39 )
        {
          v46 = a2[6];
          sub_1623A60(&v46, v39, 2);
          v40 = v16 + 48;
          if ( *(_QWORD *)(v16 + 48) )
          {
            sub_161E7C0(v16 + 48);
            v40 = v16 + 48;
          }
          v41 = v46;
          *(_QWORD *)(v16 + 48) = v46;
          if ( v41 )
            sub_1623210(&v46, v41, v40);
        }
      }
      else
      {
        v16 = sub_15A4A70(v16, v12);
      }
    }
    v23 = sub_127B390();
    v24 = sub_1644900(a2[5], v23);
    v50 = 257;
    v25 = sub_12AA3B0(a2 + 6, 0x2Du, v16, v24, (__int64)v49);
    *(_BYTE *)(a1 + 12) &= ~1u;
    *(_QWORD *)a1 = v25;
    *(_DWORD *)(a1 + 8) = 0;
    *(_DWORD *)(a1 + 16) = 0;
  }
  return a1;
}
