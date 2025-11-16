// Function: sub_1FD8980
// Address: 0x1fd8980
//
__int64 __fastcall sub_1FD8980(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  unsigned int v6; // r14d
  int v8; // eax
  unsigned int v9; // ebx
  __int64 v10; // r8
  __int64 (*v11)(); // rax
  unsigned __int32 v12; // r15d
  bool v14; // zf
  __int64 v15; // rax
  __int64 (*v16)(); // rax
  unsigned int v17; // eax
  char v18; // di
  unsigned int v19; // eax
  unsigned int v20; // edx
  __int64 (*v21)(); // rax
  __int64 (*v22)(); // rax
  __int64 v23; // r14
  __int64 v24; // rsi
  __int64 **v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rcx
  __int64 v31; // r8
  __int64 v32; // rdi
  __int64 (__fastcall *v33)(__int64, unsigned __int8); // rax
  __int64 v34; // rsi
  __int64 *v35; // rax
  __int64 v36; // rax
  __int64 v37; // rdx
  __int64 v38; // rcx
  __int64 v39; // r8
  unsigned int v40; // eax
  __int64 (*v41)(); // r10
  char v42; // [rsp+Fh] [rbp-51h] BYREF
  __int64 v43; // [rsp+10h] [rbp-50h] BYREF
  __int64 v44; // [rsp+18h] [rbp-48h]
  __int64 v45; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v46; // [rsp+28h] [rbp-38h]
  char v47; // [rsp+2Ch] [rbp-34h]

  v6 = a3;
  v8 = *(unsigned __int8 *)(a2 + 16);
  if ( (_BYTE)v8 != 13 )
  {
    if ( (_BYTE)v8 == 53 )
    {
      v22 = *(__int64 (**)())(*a1 + 112);
      if ( v22 == sub_1FD3470 )
        return 0;
      return v22();
    }
    if ( (_BYTE)v8 == 15 )
    {
      v23 = a1[12];
      v24 = sub_16498A0(a2);
      v25 = (__int64 **)sub_15A9620(v23, v24, 0);
      v28 = sub_15A06D0(v25, v24, v26, v27);
      return sub_1FD8F60(a1, v28, v29, v30, v31);
    }
    if ( (_BYTE)v8 != 14 )
    {
      if ( (_BYTE)v8 == 5 )
      {
        v20 = *(unsigned __int16 *)(a2 + 18);
      }
      else
      {
        if ( (unsigned __int8)v8 <= 0x17u )
        {
          if ( (_BYTE)v8 == 9 )
          {
            v32 = a1[14];
            v33 = *(__int64 (__fastcall **)(__int64, unsigned __int8))(*(_QWORD *)v32 + 288LL);
            if ( v33 == sub_1D45FB0 )
            {
              a3 = (unsigned __int8)a3;
              v34 = *(_QWORD *)(v32 + 8LL * (unsigned __int8)a3 + 120);
            }
            else
            {
              v34 = v33(v32, a3);
            }
            v12 = sub_1FD3F10((__int64)a1, v34, a3, a4, a5, a6);
            sub_1FD3890(
              *(_QWORD *)(a1[5] + 784),
              *(__int64 **)(a1[5] + 792),
              a1 + 10,
              *(_QWORD *)(a1[13] + 8) + 576LL,
              v12);
            return v12;
          }
          return 0;
        }
        v20 = v8 - 24;
      }
      if ( !sub_1FD8350(a1, a2, v20, a4)
        && (*(_BYTE *)(a2 + 16) <= 0x17u || !(*(unsigned __int8 (__fastcall **)(__int64 *, __int64))(*a1 + 24))(a1, a2)) )
      {
        return 0;
      }
      return sub_1FD4C00((__int64)a1, a2);
    }
    v14 = !sub_1593BB0(a2, a2, a3, a4);
    v15 = *a1;
    if ( v14 )
    {
      v21 = *(__int64 (**)())(v15 + 96);
      if ( v21 == sub_1FD34F0 )
        goto LABEL_14;
      v12 = ((__int64 (__fastcall *)(__int64 *, _QWORD, _QWORD, __int64, __int64))v21)(a1, v6, v6, 11, a2);
    }
    else
    {
      v16 = *(__int64 (**)())(v15 + 120);
      if ( v16 == sub_1FD3480 )
        goto LABEL_14;
      v12 = ((__int64 (__fastcall *)(__int64 *, __int64))v16)(a1, a2);
    }
    if ( v12 )
      return v12;
LABEL_14:
    v17 = 8 * sub_15A9520(a1[12], 0);
    if ( v17 == 32 )
    {
      v18 = 5;
    }
    else if ( v17 > 0x20 )
    {
      if ( v17 == 64 )
      {
        v18 = 6;
      }
      else
      {
        if ( v17 != 128 )
          goto LABEL_18;
        v18 = 7;
      }
    }
    else if ( v17 == 8 )
    {
      v18 = 3;
    }
    else
    {
      v18 = 4;
      if ( v17 != 16 )
      {
LABEL_18:
        v43 = 0;
        v44 = 0;
        v19 = sub_1F58D40((__int64)&v43);
LABEL_30:
        v46 = v19;
        if ( v19 <= 0x40 )
          v45 = 0;
        else
          sub_16A4EF0((__int64)&v45, 0, 0);
        v47 = 0;
        sub_169E1A0(a2 + 24, (__int64)&v45, 3u, &v42);
        if ( v42
          && (v35 = (__int64 *)sub_16498A0(a2),
              v36 = sub_159C0E0(v35, (__int64)&v45),
              (v40 = sub_1FD8F60(a1, v36, v37, v38, v39)) != 0)
          && (v41 = *(__int64 (**)())(*a1 + 64), v41 != sub_1FD34C0) )
        {
          v12 = ((__int64 (__fastcall *)(__int64 *, _QWORD, _QWORD, __int64, _QWORD, _QWORD))v41)(
                  a1,
                  (unsigned __int8)v43,
                  v6,
                  146,
                  v40,
                  0);
        }
        else
        {
          v12 = 0;
        }
        if ( v46 > 0x40 )
        {
          if ( v45 )
            j_j___libc_free_0_0(v45);
        }
        return v12;
      }
    }
    LOBYTE(v43) = v18;
    v44 = 0;
    v19 = sub_1FD3510(v18);
    goto LABEL_30;
  }
  v9 = *(_DWORD *)(a2 + 32);
  if ( v9 > 0x40 )
  {
    if ( v9 - (unsigned int)sub_16A57B0(a2 + 24) > 0x40 )
      return 0;
    v11 = *(__int64 (**)())(*a1 + 88);
    v10 = **(_QWORD **)(a2 + 24);
  }
  else
  {
    v10 = *(_QWORD *)(a2 + 24);
    v11 = *(__int64 (**)())(*a1 + 88);
  }
  if ( v11 == sub_1FD34E0 )
    return 0;
  return ((__int64 (__fastcall *)(__int64 *, _QWORD, _QWORD, __int64, __int64))v11)(a1, v6, v6, 10, v10);
}
