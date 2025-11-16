// Function: sub_1FDBD10
// Address: 0x1fdbd10
//
__int64 __fastcall sub_1FDBD10(__int64 *a1, __int64 a2)
{
  __int64 v4; // rdx
  __int64 **v5; // rax
  __int64 *v6; // r8
  __int64 v7; // rsi
  unsigned int v8; // eax
  unsigned int v9; // r14d
  unsigned int v10; // eax
  unsigned int v11; // ebx
  unsigned int v14; // eax
  __int64 v15; // rcx
  bool v16; // r10
  __int64 *v17; // rax
  unsigned __int32 v18; // r11d
  __int64 *v19; // rax
  bool v20; // al
  unsigned __int32 v21; // r11d
  __int64 v22; // rdx
  unsigned __int8 v23; // r10
  __int64 v24; // r9
  __int64 (*v25)(); // rax
  unsigned __int32 v26; // eax
  unsigned __int32 v27; // r15d
  __int64 v28; // rdi
  __int64 (__fastcall *v29)(__int64, unsigned __int8); // rcx
  __int64 (__fastcall *v30)(__int64, unsigned __int8); // rax
  __int64 v31; // r8
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v35; // rax
  unsigned __int8 v36; // [rsp+8h] [rbp-58h]
  bool v37; // [rsp+8h] [rbp-58h]
  __int64 v38; // [rsp+8h] [rbp-58h]
  bool v39; // [rsp+10h] [rbp-50h]
  unsigned __int8 v40; // [rsp+10h] [rbp-50h]
  unsigned __int8 v41; // [rsp+10h] [rbp-50h]
  unsigned __int8 v42; // [rsp+10h] [rbp-50h]
  unsigned __int8 v43; // [rsp+1Bh] [rbp-45h]
  bool v44; // [rsp+1Ch] [rbp-44h]
  unsigned __int32 v45; // [rsp+1Ch] [rbp-44h]
  unsigned __int8 v46; // [rsp+1Ch] [rbp-44h]
  unsigned __int8 v47; // [rsp+1Ch] [rbp-44h]
  __int32 v48; // [rsp+1Ch] [rbp-44h]
  unsigned __int32 v49; // [rsp+1Ch] [rbp-44h]
  __int64 v50[8]; // [rsp+20h] [rbp-40h] BYREF

  v4 = *(_QWORD *)a2;
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
  {
    v5 = *(__int64 ***)(a2 - 8);
    v6 = *v5;
    v7 = **v5;
    if ( v4 != v7 )
    {
LABEL_3:
      LOBYTE(v8) = sub_1FD35E0(a1[12], v7);
      v9 = v8;
      LOBYTE(v10) = sub_1FD35E0(a1[12], *(_QWORD *)a2);
      v11 = v10;
      if ( (_BYTE)v10 != 1 && (unsigned __int8)v9 > 1u )
      {
        v15 = a1[14];
        v16 = (_BYTE)v10 != 0 && *(_QWORD *)(v15 + 8LL * (unsigned __int8)v9 + 120) != 0;
        if ( v16 )
        {
          if ( *(_QWORD *)(v15 + 8LL * (unsigned __int8)v10 + 120) )
          {
            v17 = (*(_BYTE *)(a2 + 23) & 0x40) != 0
                ? *(__int64 **)(a2 - 8)
                : (__int64 *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
            v44 = v16;
            v18 = sub_1FD8F60(a1, *v17);
            if ( v18 )
            {
              v19 = (*(_BYTE *)(a2 + 23) & 0x40) != 0
                  ? *(__int64 **)(a2 - 8)
                  : (__int64 *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
              v39 = v44;
              v45 = v18;
              v20 = sub_1FD4DC0((__int64)a1, *v19);
              v21 = v45;
              v22 = (unsigned __int8)v9;
              v23 = v39;
              v24 = v20;
              if ( (_BYTE)v11 == (_BYTE)v9 )
              {
                v28 = a1[14];
                v29 = sub_1D45FB0;
                v30 = *(__int64 (__fastcall **)(__int64, unsigned __int8))(*(_QWORD *)v28 + 288LL);
                if ( v30 == sub_1D45FB0 )
                {
                  v31 = *(_QWORD *)(v28 + 8LL * (unsigned __int8)v9 + 120);
                }
                else
                {
                  v37 = v39;
                  v41 = v24;
                  v35 = ((__int64 (__fastcall *)(__int64, _QWORD, _QWORD))v30)(v28, v11, (unsigned __int8)v9);
                  v28 = a1[14];
                  v29 = sub_1D45FB0;
                  v31 = v35;
                  v23 = v37;
                  v24 = v41;
                  v21 = v45;
                  v30 = *(__int64 (__fastcall **)(__int64, unsigned __int8))(*(_QWORD *)v28 + 288LL);
                }
                if ( v30 == sub_1D45FB0 )
                {
                  v32 = *(_QWORD *)(v28 + 8LL * (unsigned __int8)v11 + 120);
                }
                else
                {
                  v43 = v23;
                  v38 = v31;
                  v42 = v24;
                  v49 = v21;
                  v32 = ((__int64 (__fastcall *)(__int64, _QWORD, __int64))v30)(v28, v11, v22);
                  v23 = v43;
                  v31 = v38;
                  v24 = v42;
                  v21 = v49;
                }
                if ( v31 == v32 )
                {
                  v36 = v23;
                  v40 = v24;
                  v48 = v21;
                  v27 = sub_1FD3F10((__int64)a1, v31, v22, (__int64)v29, v31, v24);
                  v33 = sub_1FD3890(
                          *(_QWORD *)(a1[5] + 784),
                          *(__int64 **)(a1[5] + 792),
                          a1 + 10,
                          *(_QWORD *)(a1[13] + 8) + 960LL,
                          v27);
                  v50[1] = v34;
                  v50[0] = v33;
                  sub_1FD3790(v50, v48, 0, 0);
                  v21 = v48;
                  v24 = v40;
                  v23 = v36;
                  if ( v27 )
                    goto LABEL_20;
                }
              }
              v25 = *(__int64 (**)())(*a1 + 64);
              if ( v25 != sub_1FD34C0 )
              {
                v46 = v23;
                v26 = ((__int64 (__fastcall *)(__int64 *, _QWORD, _QWORD, __int64, _QWORD, __int64))v25)(
                        a1,
                        v9,
                        v11,
                        158,
                        v21,
                        v24);
                v23 = v46;
                v27 = v26;
                if ( v26 )
                {
LABEL_20:
                  v47 = v23;
                  sub_1FD5CC0((__int64)a1, a2, v27, 1);
                  return v47;
                }
              }
            }
          }
        }
      }
      return 0;
    }
  }
  else
  {
    v6 = *(__int64 **)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
    v7 = *v6;
    if ( v4 != *v6 )
      goto LABEL_3;
  }
  v14 = sub_1FD8F60(a1, (__int64)v6);
  if ( v14 )
  {
    sub_1FD5CC0((__int64)a1, a2, v14, 1);
    return 1;
  }
  return 0;
}
