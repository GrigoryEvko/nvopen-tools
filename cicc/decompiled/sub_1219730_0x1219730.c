// Function: sub_1219730
// Address: 0x1219730
//
__int64 __fastcall sub_1219730(__int64 a1, _QWORD *a2)
{
  __int64 v2; // r12
  unsigned int v3; // r15d
  bool v5; // zf
  unsigned __int64 v6; // rsi
  __int64 v7; // r9
  __int64 v8; // rax
  __int64 *v9; // r8
  unsigned __int64 v10; // rdx
  int v11; // eax
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rax
  int v15; // r15d
  unsigned __int64 v16; // rdx
  char v17; // al
  __int64 v18; // rdx
  int v19; // eax
  __int64 *v20; // [rsp+8h] [rbp-168h]
  __int64 v22; // [rsp+30h] [rbp-140h]
  __int64 v23; // [rsp+38h] [rbp-138h] BYREF
  __int64 v24; // [rsp+40h] [rbp-130h] BYREF
  char v25; // [rsp+48h] [rbp-128h]
  _QWORD *v26; // [rsp+50h] [rbp-120h] BYREF
  __int64 v27; // [rsp+58h] [rbp-118h]
  _QWORD v28[2]; // [rsp+60h] [rbp-110h] BYREF
  __int64 *v29[2]; // [rsp+70h] [rbp-100h] BYREF
  __int64 v30; // [rsp+80h] [rbp-F0h] BYREF
  int v31[8]; // [rsp+90h] [rbp-E0h] BYREF
  __int16 v32; // [rsp+B0h] [rbp-C0h]
  _QWORD *v33; // [rsp+C0h] [rbp-B0h] BYREF
  __int64 v34; // [rsp+C8h] [rbp-A8h]
  _BYTE v35[48]; // [rsp+D0h] [rbp-A0h] BYREF
  _DWORD *v36; // [rsp+100h] [rbp-70h] BYREF
  __int64 v37; // [rsp+108h] [rbp-68h]
  _BYTE v38[96]; // [rsp+110h] [rbp-60h] BYREF

  v2 = a1 + 176;
  *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
  v26 = v28;
  v27 = 0;
  LOBYTE(v28[0]) = 0;
  if ( (unsigned __int8)sub_120AFE0(a1, 12, "expected '(' in target extension type")
    || (v3 = sub_120B3D0(a1, (__int64)&v26), (_BYTE)v3) )
  {
    v3 = 1;
    goto LABEL_3;
  }
  v5 = *(_DWORD *)(a1 + 240) == 4;
  v33 = v35;
  v34 = 0x600000000LL;
  v36 = v38;
  v37 = 0xC00000000LL;
  if ( v5 )
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v11 = sub_1205200(v2);
        *(_DWORD *)(a1 + 240) = v11;
        if ( v11 != 529 )
          break;
        v6 = (unsigned __int64)v31;
        v3 = sub_120BD00(a1, v31);
        if ( (_BYTE)v3 )
          goto LABEL_30;
        v14 = (unsigned int)v37;
        v15 = v31[0];
        v16 = (unsigned int)v37 + 1LL;
        if ( v16 > HIDWORD(v37) )
        {
          sub_C8D5F0((__int64)&v36, v38, v16, 4u, v12, v13);
          v14 = (unsigned int)v37;
        }
        v36[v14] = v15;
        v3 = 1;
        LODWORD(v37) = v37 + 1;
        if ( *(_DWORD *)(a1 + 240) != 4 )
          goto LABEL_19;
      }
      HIBYTE(v32) = 1;
      if ( (_BYTE)v3 )
        break;
      v6 = (unsigned __int64)v29;
      *(_QWORD *)v31 = "expected type";
      LOBYTE(v32) = 3;
      v3 = sub_12190A0(a1, v29, v31, 1);
      if ( (_BYTE)v3 )
        goto LABEL_30;
      v8 = (unsigned int)v34;
      v9 = v29[0];
      v10 = (unsigned int)v34 + 1LL;
      if ( v10 > HIDWORD(v34) )
      {
        v20 = v29[0];
        sub_C8D5F0((__int64)&v33, v35, v10, 8u, (__int64)v29[0], v7);
        v8 = (unsigned int)v34;
        v9 = v20;
      }
      v33[v8] = v9;
      LODWORD(v34) = v34 + 1;
      if ( *(_DWORD *)(a1 + 240) != 4 )
        goto LABEL_19;
    }
    v6 = *(_QWORD *)(a1 + 232);
    *(_QWORD *)v31 = "expected uint32 param";
    LOBYTE(v32) = 3;
    sub_11FD800(v2, v6, (__int64)v31, 1);
    goto LABEL_30;
  }
LABEL_19:
  v6 = 13;
  v3 = sub_120AFE0(a1, 13, "expected ')' in target extension type");
  if ( !(_BYTE)v3 )
  {
    v6 = *(_QWORD *)a1;
    sub_BCFB10((__int64)&v24, *(_QWORD **)a1, (__int64)v26, v27, v33, (unsigned int)v34, v36, (unsigned int)v37);
    v17 = v25;
    v25 &= ~2u;
    v18 = v24;
    v19 = v17 & 1;
    if ( v19 )
    {
      v24 = 0;
      v22 = v18 | 1;
      if ( (v18 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      {
        v23 = v18 | 1;
        v22 = 0;
        sub_C64870((__int64)v29, &v23);
        v6 = *(_QWORD *)(a1 + 232);
        v32 = 260;
        *(_QWORD *)v31 = v29;
        sub_11FD800(v2, v6, (__int64)v31, 1);
        if ( v29[0] != &v30 )
        {
          v6 = v30 + 1;
          j_j___libc_free_0(v29[0], v30 + 1);
        }
        if ( (v23 & 1) != 0 || (v23 & 0xFFFFFFFFFFFFFFFELL) != 0 )
          sub_C63C30(&v23, v6);
        v3 = (v22 & 1) == 0;
        LOBYTE(v19) = v25 & 1;
        if ( (v25 & 2) != 0 )
          sub_9D22F0(&v24);
        goto LABEL_27;
      }
      v18 = 0;
    }
    *a2 = v18;
LABEL_27:
    if ( (_BYTE)v19 && v24 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v24 + 8LL))(v24);
  }
LABEL_30:
  if ( v36 != (_DWORD *)v38 )
    _libc_free(v36, v6);
  if ( v33 != (_QWORD *)v35 )
    _libc_free(v33, v6);
LABEL_3:
  if ( v26 != v28 )
    j_j___libc_free_0(v26, v28[0] + 1LL);
  return v3;
}
