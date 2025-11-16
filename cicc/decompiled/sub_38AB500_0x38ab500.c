// Function: sub_38AB500
// Address: 0x38ab500
//
__int64 __fastcall sub_38AB500(__int64 a1, __int64 *a2, __int64 *a3, double a4, double a5, double a6)
{
  unsigned __int64 v8; // r13
  unsigned int v9; // r14d
  bool v11; // zf
  int v12; // r9d
  char v13; // dl
  __int64 v14; // r10
  __int64 v15; // r8
  __int64 v16; // rax
  __int64 *v17; // rax
  _QWORD *v18; // rax
  _QWORD *v19; // rdi
  _QWORD *v20; // rcx
  const char *v21; // rax
  int v22; // eax
  __int64 v23; // rdx
  int v24; // ebx
  __int64 v25; // r13
  __int64 v26; // rax
  __int64 v27; // rcx
  __int64 v28; // r8
  __int64 v29; // r9
  __int64 v30; // r15
  __int64 v31; // rbx
  __int64 v32; // r13
  __int64 *v33; // rax
  __int64 v34; // [rsp+0h] [rbp-3E0h]
  unsigned __int64 v35; // [rsp+10h] [rbp-3D0h]
  __int64 v36; // [rsp+10h] [rbp-3D0h]
  __int64 v37; // [rsp+18h] [rbp-3C8h]
  unsigned __int64 v38; // [rsp+28h] [rbp-3B8h] BYREF
  __int64 v39; // [rsp+30h] [rbp-3B0h] BYREF
  __int64 v40; // [rsp+38h] [rbp-3A8h] BYREF
  __int64 v41; // [rsp+40h] [rbp-3A0h] BYREF
  __int64 v42; // [rsp+48h] [rbp-398h] BYREF
  unsigned __int64 v43[2]; // [rsp+50h] [rbp-390h] BYREF
  char v44; // [rsp+60h] [rbp-380h]
  char v45; // [rsp+61h] [rbp-37Fh]
  __int64 v46; // [rsp+70h] [rbp-370h] BYREF
  _BYTE *v47; // [rsp+78h] [rbp-368h]
  _BYTE *v48; // [rsp+80h] [rbp-360h]
  __int64 v49; // [rsp+88h] [rbp-358h]
  int v50; // [rsp+90h] [rbp-350h]
  _BYTE v51[264]; // [rsp+98h] [rbp-348h] BYREF
  const char *v52; // [rsp+1A0h] [rbp-240h] BYREF
  __int64 v53; // [rsp+1A8h] [rbp-238h]
  _BYTE v54[560]; // [rsp+1B0h] [rbp-230h] BYREF

  v8 = *(_QWORD *)(a1 + 56);
  v38 = 0;
  if ( (unsigned __int8)sub_38AB270((__int64 **)a1, &v39, a3, a4, a5, a6) )
    return 1;
  if ( (unsigned __int8)sub_388AF10(a1, 4, "expected ',' after switch condition") )
    return 1;
  if ( (unsigned __int8)sub_38AB2F0(a1, &v40, &v38, a3, a4, a5, a6) )
    return 1;
  v9 = sub_388AF10(a1, 6, "expected '[' with switch table");
  if ( (_BYTE)v9 )
  {
    return 1;
  }
  else if ( *(_BYTE *)(*(_QWORD *)v39 + 8LL) == 11 )
  {
    v11 = *(_DWORD *)(a1 + 64) == 7;
    v46 = 0;
    v47 = v51;
    v48 = v51;
    v52 = v54;
    v49 = 32;
    v50 = 0;
    v53 = 0x2000000000LL;
    if ( v11 )
    {
LABEL_37:
      v22 = sub_3887100(a1 + 8);
      v23 = v40;
      v24 = v53;
      *(_DWORD *)(a1 + 64) = v22;
      v25 = v39;
      v37 = v23;
      v26 = sub_1648B60(64);
      v30 = v26;
      if ( v26 )
        sub_15FFAB0(v26, v25, v37, v24, 0);
      v31 = 0;
      v32 = 16LL * (unsigned int)v53;
      if ( (_DWORD)v53 )
      {
        do
        {
          v33 = (__int64 *)&v52[v31];
          v31 += 16;
          sub_15FFFB0(v30, *v33, v33[1], v27, v28, v29);
        }
        while ( v31 != v32 );
      }
      *a2 = v30;
    }
    else
    {
      while ( 1 )
      {
        v35 = *(_QWORD *)(a1 + 56);
        if ( (unsigned __int8)sub_38AB270((__int64 **)a1, &v41, a3, a4, a5, a6) )
          break;
        if ( (unsigned __int8)sub_388AF10(a1, 4, "expected ',' after case value") )
          break;
        v43[0] = 0;
        if ( (unsigned __int8)sub_38AB2F0(a1, &v42, v43, a3, a4, a5, a6) )
          break;
        v18 = v47;
        if ( v48 != v47 )
          goto LABEL_10;
        v19 = &v47[8 * HIDWORD(v49)];
        if ( v47 != (_BYTE *)v19 )
        {
          v20 = 0;
          while ( v41 != *v18 )
          {
            if ( *v18 == -2 )
              v20 = v18;
            if ( v19 == ++v18 )
            {
              if ( !v20 )
                goto LABEL_34;
              *v20 = v41;
              --v50;
              ++v46;
              goto LABEL_11;
            }
          }
LABEL_25:
          v45 = 1;
          v21 = "duplicate case value in switch";
LABEL_26:
          v43[0] = (unsigned __int64)v21;
          v44 = 3;
          v9 = sub_38814C0(a1 + 8, v35, (__int64)v43);
          goto LABEL_27;
        }
LABEL_34:
        if ( HIDWORD(v49) < (unsigned int)v49 )
        {
          ++HIDWORD(v49);
          *v19 = v41;
          ++v46;
        }
        else
        {
LABEL_10:
          sub_16CCBA0((__int64)&v46, v41);
          if ( !v13 )
            goto LABEL_25;
        }
LABEL_11:
        v14 = v41;
        if ( *(_BYTE *)(v41 + 16) != 13 )
        {
          v45 = 1;
          v21 = "case value is not a constant integer";
          goto LABEL_26;
        }
        v15 = v42;
        v16 = (unsigned int)v53;
        if ( (unsigned int)v53 >= HIDWORD(v53) )
        {
          v34 = v42;
          v36 = v41;
          sub_16CD150((__int64)&v52, v54, 0, 16, v42, v12);
          v16 = (unsigned int)v53;
          v15 = v34;
          v14 = v36;
        }
        v17 = (__int64 *)&v52[16 * v16];
        *v17 = v14;
        v17[1] = v15;
        LODWORD(v53) = v53 + 1;
        if ( *(_DWORD *)(a1 + 64) == 7 )
          goto LABEL_37;
      }
      v9 = 1;
    }
LABEL_27:
    if ( v52 != v54 )
      _libc_free((unsigned __int64)v52);
    if ( v48 != v47 )
      _libc_free((unsigned __int64)v48);
  }
  else
  {
    v54[1] = 1;
    v52 = "switch condition must have integer type";
    v54[0] = 3;
    return (unsigned int)sub_38814C0(a1 + 8, v8, (__int64)&v52);
  }
  return v9;
}
