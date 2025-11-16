// Function: sub_148B860
// Address: 0x148b860
//
__int64 __fastcall sub_148B860(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // r11d
  __int64 v7; // rbx
  __int64 v10; // r9
  char v11; // al
  int v12; // eax
  __int64 v13; // rax
  unsigned int v14; // eax
  unsigned int v15; // eax
  __int64 v16; // r9
  unsigned __int8 v17; // r11
  unsigned __int8 v18; // al
  __int64 v19; // r9
  __int64 v20; // rax
  __int64 v21; // [rsp-C8h] [rbp-C8h]
  __int64 v22; // [rsp-C8h] [rbp-C8h]
  __int64 v23; // [rsp-C0h] [rbp-C0h]
  char v24; // [rsp-C0h] [rbp-C0h]
  unsigned int v26; // [rsp-B0h] [rbp-B0h]
  __int64 v27; // [rsp-B0h] [rbp-B0h]
  __int64 v28; // [rsp-B0h] [rbp-B0h]
  unsigned __int8 v29; // [rsp-B0h] [rbp-B0h]
  unsigned __int8 v30; // [rsp-B0h] [rbp-B0h]
  unsigned __int8 v31; // [rsp-B0h] [rbp-B0h]
  unsigned __int64 v32; // [rsp-A8h] [rbp-A8h] BYREF
  int v33; // [rsp-A0h] [rbp-A0h]
  unsigned __int64 v34; // [rsp-98h] [rbp-98h] BYREF
  unsigned int v35; // [rsp-90h] [rbp-90h]
  unsigned __int64 v36; // [rsp-88h] [rbp-88h] BYREF
  unsigned int v37; // [rsp-80h] [rbp-80h]
  __int64 v38; // [rsp-78h] [rbp-78h] BYREF
  unsigned int v39; // [rsp-70h] [rbp-70h]
  unsigned __int8 v40; // [rsp-68h] [rbp-68h]
  __int64 v41[2]; // [rsp-58h] [rbp-58h] BYREF
  char v42; // [rsp-48h] [rbp-48h]

  v6 = 0;
  if ( ((a2 - 36) & 0xFFFFFFFB) == 0 && *(_WORD *)(a3 + 24) == 7 && *(_WORD *)(a5 + 24) == 7 )
  {
    v7 = *(_QWORD *)(a5 + 48);
    if ( v7 != *(_QWORD *)(a3 + 48) )
      return v6;
    sub_1478E30((__int64)&v38, a1, a3, a5);
    sub_1478E30((__int64)v41, a1, a4, a6);
    v6 = v40;
    if ( v40 )
    {
      LOBYTE(v6) = v42;
      if ( !v42 )
        goto LABEL_20;
      v10 = a6;
      if ( v39 <= 0x40 )
      {
        if ( v38 != v41[0] )
        {
          LOBYTE(v6) = 0;
          goto LABEL_22;
        }
        if ( !v38 )
          goto LABEL_22;
      }
      else
      {
        v23 = a6;
        v26 = v39;
        v11 = sub_16A5220(&v38, v41);
        LOBYTE(v6) = v11;
        if ( !v11 )
          goto LABEL_22;
        v21 = v23;
        v24 = v11;
        v12 = sub_16A57B0(&v38);
        LOBYTE(v6) = v24;
        v10 = v21;
        if ( v26 == v12 )
          goto LABEL_22;
      }
      v33 = 1;
      v32 = 0;
      v27 = v10;
      if ( a2 == 36 )
      {
        sub_13A38D0((__int64)&v36, (__int64)v41);
        v19 = v27;
        if ( v37 > 0x40 )
        {
          sub_16A8F40(&v36);
          v19 = v27;
        }
        else
        {
          v36 = ~v36 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v37);
        }
        v22 = v19;
        sub_16A7400(&v36);
        v35 = v37;
        v34 = v36;
        v37 = 0;
        sub_14536D0((__int64 *)&v32, (__int64 *)&v34);
        sub_135E100((__int64 *)&v34);
        sub_135E100((__int64 *)&v36);
        v16 = v22;
      }
      else
      {
        v13 = sub_1456040(a4);
        v14 = sub_1456C90(a1, v13);
        sub_13D00B0((__int64)&v36, v14);
        sub_16A7590(&v36, v41);
        v15 = v37;
        v37 = 0;
        v34 = v36;
        v32 = v36;
        v33 = v15;
        v35 = 0;
        sub_135E100((__int64 *)&v34);
        sub_135E100((__int64 *)&v36);
        v16 = v27;
      }
      v28 = v16;
      v17 = sub_146D950(a1, v16, v7);
      if ( v17 )
      {
        v20 = sub_145CF40(a1, (__int64)&v32);
        v17 = sub_148B410(a1, v7, a2, v28, v20);
      }
      v29 = v17;
      sub_135E100((__int64 *)&v32);
      v6 = v29;
    }
    if ( !v42 )
    {
      v18 = v40;
      goto LABEL_19;
    }
LABEL_22:
    v31 = v6;
    sub_135E100(v41);
    v18 = v40;
    v6 = v31;
LABEL_19:
    if ( !v18 )
      return v6;
LABEL_20:
    v30 = v6;
    sub_135E100(&v38);
    return v30;
  }
  return 0;
}
