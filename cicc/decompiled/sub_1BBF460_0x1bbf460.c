// Function: sub_1BBF460
// Address: 0x1bbf460
//
__int64 __fastcall sub_1BBF460(__int64 a1, __int64 *a2, __int64 *a3, double a4, double a5, double a6)
{
  __int64 v6; // r11
  __int64 v9; // r10
  __int64 v10; // r13
  _BYTE *v11; // rdx
  unsigned __int16 v12; // si
  __int64 v13; // rcx
  __int64 v14; // rax
  _QWORD *v16; // rax
  _QWORD *v17; // r12
  __int64 v18; // rbx
  __int64 v19; // rdx
  unsigned __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rdx
  unsigned __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rdx
  unsigned __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rdi
  unsigned __int64 *v29; // r13
  __int64 v30; // rax
  unsigned __int64 v31; // rcx
  __int64 v32; // rsi
  __int64 v33; // rsi
  unsigned __int8 *v34; // rsi
  bool v35; // zf
  __int64 v36; // rcx
  _BYTE *v37; // rdx
  unsigned __int16 v38; // si
  __int64 v39; // rax
  __int64 v40; // [rsp+8h] [rbp-68h]
  __int64 v41; // [rsp+10h] [rbp-60h]
  __int64 v42; // [rsp+10h] [rbp-60h]
  __int64 v43; // [rsp+18h] [rbp-58h]
  _QWORD *v44; // [rsp+18h] [rbp-58h]
  __int64 v45[2]; // [rsp+20h] [rbp-50h] BYREF
  __int16 v46; // [rsp+30h] [rbp-40h]

  v9 = *(_QWORD *)(a1 + 16);
  v10 = *(_QWORD *)(a1 + 8);
  switch ( *(_DWORD *)(a1 + 24) )
  {
    case 1:
      return sub_1904E90((__int64)a2, *(_DWORD *)a1, *(_QWORD *)(a1 + 8), v9, a3, 0, a4, a5, a6);
    case 2:
      v35 = *(_DWORD *)a1 == 51;
      v46 = 257;
      if ( !v35 )
      {
        v36 = v9;
        v37 = (_BYTE *)v10;
        v38 = 4;
        goto LABEL_38;
      }
      v13 = v9;
      v11 = (_BYTE *)v10;
      v12 = 40;
      goto LABEL_3;
    case 3:
      v11 = *(_BYTE **)(a1 + 8);
      v12 = 36;
      v46 = 257;
      v13 = v9;
      goto LABEL_3;
    case 4:
      v35 = *(_DWORD *)a1 == 51;
      v46 = 257;
      if ( v35 )
      {
        v13 = v9;
        v11 = (_BYTE *)v10;
        v12 = 38;
LABEL_3:
        v14 = sub_12AA0C0(a2, v12, v11, v13, (__int64)v45);
        v9 = *(_QWORD *)(a1 + 16);
        v10 = *(_QWORD *)(a1 + 8);
        v6 = v14;
      }
      else
      {
        v36 = v9;
        v37 = (_BYTE *)v10;
        v38 = 2;
LABEL_38:
        v39 = sub_1289B20(a2, v38, v37, v36, (__int64)v45, 0);
        v9 = *(_QWORD *)(a1 + 16);
        v10 = *(_QWORD *)(a1 + 8);
        v6 = v39;
      }
LABEL_4:
      if ( *(_BYTE *)(v6 + 16) <= 0x10u && *(_BYTE *)(v10 + 16) <= 0x10u && *(_BYTE *)(v9 + 16) <= 0x10u )
        return sub_15A2DC0(v6, (__int64 *)v10, v9, 0);
      v41 = v9;
      v43 = v6;
      v46 = 257;
      v16 = sub_1648A60(56, 3u);
      v17 = v16;
      if ( v16 )
      {
        v40 = v41;
        v18 = (__int64)v16;
        v42 = v43;
        v44 = v16 - 9;
        sub_15F1EA0((__int64)v16, *(_QWORD *)v10, 55, (__int64)(v16 - 9), 3, 0);
        if ( *(v17 - 9) )
        {
          v19 = *(v17 - 8);
          v20 = *(v17 - 7) & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v20 = v19;
          if ( v19 )
            *(_QWORD *)(v19 + 16) = *(_QWORD *)(v19 + 16) & 3LL | v20;
        }
        *(v17 - 9) = v42;
        v21 = *(_QWORD *)(v42 + 8);
        *(v17 - 8) = v21;
        if ( v21 )
          *(_QWORD *)(v21 + 16) = (unsigned __int64)(v17 - 8) | *(_QWORD *)(v21 + 16) & 3LL;
        *(v17 - 7) = (v42 + 8) | *(v17 - 7) & 3LL;
        *(_QWORD *)(v42 + 8) = v44;
        if ( *(v17 - 6) )
        {
          v22 = *(v17 - 5);
          v23 = *(v17 - 4) & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v23 = v22;
          if ( v22 )
            *(_QWORD *)(v22 + 16) = *(_QWORD *)(v22 + 16) & 3LL | v23;
        }
        *(v17 - 6) = v10;
        v24 = *(_QWORD *)(v10 + 8);
        *(v17 - 5) = v24;
        if ( v24 )
          *(_QWORD *)(v24 + 16) = (unsigned __int64)(v17 - 5) | *(_QWORD *)(v24 + 16) & 3LL;
        *(v17 - 4) = (v10 + 8) | *(v17 - 4) & 3LL;
        *(_QWORD *)(v10 + 8) = v17 - 6;
        if ( *(v17 - 3) )
        {
          v25 = *(v17 - 2);
          v26 = *(v17 - 1) & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v26 = v25;
          if ( v25 )
            *(_QWORD *)(v25 + 16) = *(_QWORD *)(v25 + 16) & 3LL | v26;
        }
        *(v17 - 3) = v40;
        if ( v40 )
        {
          v27 = *(_QWORD *)(v40 + 8);
          *(v17 - 2) = v27;
          if ( v27 )
            *(_QWORD *)(v27 + 16) = (unsigned __int64)(v17 - 2) | *(_QWORD *)(v27 + 16) & 3LL;
          *(v17 - 1) = (v40 + 8) | *(v17 - 1) & 3LL;
          *(_QWORD *)(v40 + 8) = v17 - 3;
        }
        sub_164B780((__int64)v17, v45);
      }
      else
      {
        v18 = 0;
      }
      v28 = a2[1];
      if ( v28 )
      {
        v29 = (unsigned __int64 *)a2[2];
        sub_157E9D0(v28 + 40, (__int64)v17);
        v30 = v17[3];
        v31 = *v29;
        v17[4] = v29;
        v31 &= 0xFFFFFFFFFFFFFFF8LL;
        v17[3] = v31 | v30 & 7;
        *(_QWORD *)(v31 + 8) = v17 + 3;
        *v29 = *v29 & 7 | (unsigned __int64)(v17 + 3);
      }
      sub_164B780(v18, a3);
      v32 = *a2;
      if ( *a2 )
      {
        v45[0] = *a2;
        sub_1623A60((__int64)v45, v32, 2);
        v33 = v17[6];
        if ( v33 )
          sub_161E7C0((__int64)(v17 + 6), v33);
        v34 = (unsigned __int8 *)v45[0];
        v17[6] = v45[0];
        if ( v34 )
          sub_1623210((__int64)v45, v34, (__int64)(v17 + 6));
      }
      return (__int64)v17;
    case 5:
      v13 = *(_QWORD *)(a1 + 16);
      v12 = 34;
      v46 = 257;
      v11 = (_BYTE *)v10;
      goto LABEL_3;
    default:
      goto LABEL_4;
  }
}
