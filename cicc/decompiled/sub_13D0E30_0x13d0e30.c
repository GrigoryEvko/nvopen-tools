// Function: sub_13D0E30
// Address: 0x13d0e30
//
__int64 __fastcall sub_13D0E30(__int64 a1, __int64 a2, __int64 a3, unsigned int a4, __int64 a5, __int64 a6, __int64 a7)
{
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 result; // rax
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // rax
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 v19; // rcx
  __int64 v20; // r15
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24; // rax
  unsigned __int8 v25; // al
  __int64 v26; // r12
  _QWORD *v27; // rax
  __int64 v28; // rbx
  __int64 v29; // rax
  __int64 v30; // rdi
  __int64 v31; // r12
  _QWORD *v32; // rax
  __int64 v33; // rbx
  __int64 v34; // rax
  __int64 v35; // rdi
  __int64 v36; // rdi
  __int64 v37; // r13
  __int64 v38; // rax
  __int64 v39; // r8
  __int64 v40; // r9
  unsigned __int8 v41; // al
  __int64 v42; // rax
  char v43; // al
  __int64 v44; // r9
  char v45; // al
  char v46; // al
  char v47; // al
  char v48; // al
  unsigned __int8 v49; // bl
  __int64 v50; // rax
  __int64 v51; // rax
  char v52; // al
  bool v53; // al
  bool v54; // al
  bool v55; // al
  bool v56; // al
  __int64 v57; // [rsp+10h] [rbp-100h]
  __int64 *v58; // [rsp+10h] [rbp-100h]
  __int64 *v59; // [rsp+10h] [rbp-100h]
  __int64 *v60; // [rsp+10h] [rbp-100h]
  __int64 *v61; // [rsp+10h] [rbp-100h]
  __int64 v62; // [rsp+10h] [rbp-100h]
  __int64 v63; // [rsp+10h] [rbp-100h]
  __int64 v64; // [rsp+10h] [rbp-100h]
  __int64 v65; // [rsp+18h] [rbp-F8h]
  __int64 v67; // [rsp+20h] [rbp-F0h]
  __int64 v68; // [rsp+20h] [rbp-F0h]
  unsigned __int64 v69[2]; // [rsp+28h] [rbp-E8h] BYREF
  unsigned __int16 v70; // [rsp+3Dh] [rbp-D3h]
  unsigned __int8 v71; // [rsp+3Fh] [rbp-D1h]
  __int64 *v72; // [rsp+40h] [rbp-D0h] BYREF
  __int64 v73; // [rsp+48h] [rbp-C8h]
  _BYTE v74[64]; // [rsp+50h] [rbp-C0h] BYREF
  __int64 *v75; // [rsp+90h] [rbp-80h] BYREF
  __int64 v76; // [rsp+98h] [rbp-78h]
  _BYTE v77[112]; // [rsp+A0h] [rbp-70h] BYREF

  v69[0] = sub_1649C60(a6);
  a7 = sub_1649C60(a7);
  if ( (unsigned __int8)sub_14BFF20(v69[0], a1, 0, 0, 0, 0) && *(_BYTE *)(a7 + 16) == 15 && a4 - 32 <= 1 )
  {
    v31 = (unsigned __int8)sub_15FF820(a4) ^ 1u;
    v32 = *(_QWORD **)v69[0];
    if ( *(_BYTE *)(*(_QWORD *)v69[0] + 8LL) == 16 )
    {
      v33 = v32[4];
      v34 = sub_1643320(*v32);
      v35 = sub_16463B0(v34, (unsigned int)v33);
    }
    else
    {
      v35 = sub_1643320(*v32);
    }
    return sub_15A0680(v35, v31, 0);
  }
  if ( a4 <= 0x21 )
  {
    if ( a4 <= 0x1F )
      return 0;
  }
  else
  {
    result = 0;
    if ( a4 - 34 > 3 )
      return result;
    a4 = sub_15FF420(a4);
  }
  v65 = sub_13CCBD0(a1, v69, 0, v9, v10, v11);
  v16 = sub_13CCBD0(a1, (unsigned __int64 *)&a7, 0, v13, v14, v15);
  v18 = v16;
  if ( v69[0] == a7 )
    return sub_15A35F0((unsigned __int16)a4, v65, v16, 0, v17, v16);
  v19 = a4 - 32;
  result = 0;
  if ( (unsigned int)v19 <= 1 )
  {
    if ( *(_BYTE *)(v69[0] + 16) != 53 )
      goto LABEL_9;
    v41 = *(_BYTE *)(a7 + 16);
    if ( v41 > 0x17u )
    {
      if ( v41 != 53 )
        goto LABEL_9;
    }
    else if ( v41 != 3 )
    {
      goto LABEL_9;
    }
    if ( *(_BYTE *)(v65 + 16) == 13 && *(_BYTE *)(v18 + 16) == 13 )
    {
      v62 = v18;
      v70 = 0;
      v42 = sub_15F2060(v69[0]);
      v71 = sub_15E4690(v42, 0);
      v43 = sub_140E950(v69[0], &v72, a1, a2, ((unsigned __int64)v71 << 16) | v70);
      v44 = v62;
      if ( v43 )
      {
        v52 = sub_140E950(a7, &v75, a1, a2, v70 | ((unsigned __int64)v71 << 16));
        v44 = v62;
        if ( v52 )
        {
          v53 = sub_13D0200((__int64 *)(v65 + 24), *(_DWORD *)(v65 + 32) - 1);
          v44 = v62;
          if ( !v53 )
          {
            v54 = sub_13D0200((__int64 *)(v62 + 24), *(_DWORD *)(v62 + 32) - 1);
            v44 = v62;
            if ( !v54 )
            {
              v55 = sub_13D0480(v65 + 24, (unsigned __int64)v72);
              v44 = v62;
              if ( v55 )
              {
                v56 = sub_13D0480(v62 + 24, (unsigned __int64)v75);
                v44 = v62;
                if ( v56 )
                  goto LABEL_52;
              }
            }
          }
        }
      }
    }
    else
    {
      v64 = v18;
      v51 = sub_15F2060(v69[0]);
      sub_15E4690(v51, 0);
      v44 = v64;
    }
    v63 = v44;
    v45 = sub_1642FB0(*(_QWORD *)v69[0]);
    v18 = v63;
    if ( !v45 )
    {
      v46 = sub_1642FB0(*(_QWORD *)a7);
      v18 = v63;
      if ( !v46 )
      {
        v47 = sub_1593BB0(v65);
        v18 = v63;
        if ( v47 )
        {
          v48 = sub_1593BB0(v63);
          v18 = v63;
          if ( v48 )
          {
LABEL_52:
            v49 = sub_15FF820(a4) ^ 1;
            v50 = sub_13D0DF0(*(_QWORD **)v69[0]);
            return sub_15A0680(v50, v49, 0);
          }
        }
      }
    }
LABEL_9:
    v57 = v18;
    v20 = sub_13CCBD0(a1, v69, 1, v19, v17, v18);
    v24 = sub_13CCBD0(a1, (unsigned __int64 *)&a7, 1, v21, v22, v23);
    if ( v69[0] == a7 )
    {
      v37 = sub_15A2B30(v57, v24, 0, 0);
      v38 = sub_15A2B30(v65, v20, 0, 0);
      return sub_15A35F0(a4, v38, v37, 0, v39, v40);
    }
    v75 = (__int64 *)v77;
    v73 = 0x800000000LL;
    v76 = 0x800000000LL;
    v72 = (__int64 *)v74;
    sub_14AD470(v69[0], &v72, a1, 0, 6);
    sub_14AD470(a7, &v75, a1, 0, 6);
    v58 = &v72[(unsigned int)v73];
    if ( v58 == sub_13CBC00(v72, (__int64)v58, (unsigned __int8 (__fastcall *)(_QWORD))sub_134E780)
      && (v59 = &v75[(unsigned int)v76], v59 == sub_13CC800(v75, (__int64)v59))
      || (v60 = &v75[(unsigned int)v76],
          v60 == sub_13CBC00(v75, (__int64)v60, (unsigned __int8 (__fastcall *)(_QWORD))sub_134E780))
      && (v61 = &v72[(unsigned int)v73], v61 == sub_13CC800(v72, (__int64)v61)) )
    {
      v25 = sub_15FF820(a4) ^ 1;
LABEL_13:
      v26 = v25;
      v27 = *(_QWORD **)v69[0];
      if ( *(_BYTE *)(*(_QWORD *)v69[0] + 8LL) == 16 )
      {
        v28 = v27[4];
        v29 = sub_1643320(*v27);
        v30 = sub_16463B0(v29, (unsigned int)v28);
      }
      else
      {
        v30 = sub_1643320(*v27);
      }
      result = sub_15A0680(v30, v26, 0);
      if ( v75 != (__int64 *)v77 )
      {
        v67 = result;
        _libc_free((unsigned __int64)v75);
        result = v67;
      }
      if ( v72 != (__int64 *)v74 )
      {
        v68 = result;
        _libc_free((unsigned __int64)v72);
        return v68;
      }
      return result;
    }
    if ( (unsigned __int8)sub_140B1C0(v69[0], a2, 0) && (unsigned __int8)sub_14BFF20(a7, a1, 0, 0, a5, a3) )
    {
      v36 = v69[0];
    }
    else
    {
      if ( !(unsigned __int8)sub_140B1C0(a7, a2, 0) || !(unsigned __int8)sub_14BFF20(v69[0], a1, 0, 0, a5, a3) )
      {
LABEL_36:
        if ( v75 != (__int64 *)v77 )
          _libc_free((unsigned __int64)v75);
        if ( v72 != (__int64 *)v74 )
          _libc_free((unsigned __int64)v72);
        return 0;
      }
      v36 = a7;
    }
    if ( v36 && !(unsigned __int8)sub_139D0F0(v36, 1) )
    {
      v25 = sub_15FF850(a4);
      goto LABEL_13;
    }
    goto LABEL_36;
  }
  return result;
}
