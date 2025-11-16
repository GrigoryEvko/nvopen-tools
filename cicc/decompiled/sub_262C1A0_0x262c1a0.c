// Function: sub_262C1A0
// Address: 0x262c1a0
//
__int64 __fastcall sub_262C1A0(__int64 a1, __int64 a2)
{
  bool v5; // zf
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // r14
  char *v9; // rcx
  __int64 *v10; // rsi
  __int64 v11; // r13
  _BYTE *v12; // rsi
  unsigned __int64 v13; // rcx
  unsigned __int64 v14; // rax
  __int64 *v15; // r13
  _BYTE *v16; // r14
  unsigned __int64 v17; // r15
  _QWORD *v18; // rax
  _QWORD *v19; // r10
  __int64 v20; // rcx
  __int64 v21; // rdx
  unsigned __int64 *v22; // r10
  _QWORD *v23; // rax
  _QWORD *v24; // rdx
  char v25; // di
  void (__fastcall *v26)(__int64, __int64 *); // rax
  unsigned __int64 v27; // r10
  _QWORD *v28; // rax
  unsigned __int64 v29; // rdi
  unsigned __int64 *v30; // [rsp+8h] [rbp-D8h]
  _QWORD *v31; // [rsp+10h] [rbp-D0h]
  _QWORD *v32; // [rsp+18h] [rbp-C8h]
  _QWORD *v33; // [rsp+18h] [rbp-C8h]
  __int64 v34; // [rsp+20h] [rbp-C0h]
  _QWORD *v35; // [rsp+20h] [rbp-C0h]
  _QWORD *v36; // [rsp+20h] [rbp-C0h]
  _QWORD *v37; // [rsp+28h] [rbp-B8h]
  __int64 v38; // [rsp+30h] [rbp-B0h]
  _QWORD *v39; // [rsp+30h] [rbp-B0h]
  unsigned __int64 v40; // [rsp+38h] [rbp-A8h]
  char v41; // [rsp+4Eh] [rbp-92h] BYREF
  char v42; // [rsp+4Fh] [rbp-91h] BYREF
  __int64 v43; // [rsp+50h] [rbp-90h] BYREF
  __int64 *v44; // [rsp+58h] [rbp-88h] BYREF
  __int64 *v45; // [rsp+60h] [rbp-80h] BYREF
  unsigned __int64 v46; // [rsp+68h] [rbp-78h]
  _QWORD v47[2]; // [rsp+70h] [rbp-70h] BYREF
  __int64 v48[2]; // [rsp+80h] [rbp-60h] BYREF
  _QWORD v49[2]; // [rsp+90h] [rbp-50h] BYREF
  char v50; // [rsp+A0h] [rbp-40h]
  char v51; // [rsp+A1h] [rbp-3Fh]

  (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 104LL))(a1);
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _QWORD, __int64 **, __int64 *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "TTRes",
         0,
         0,
         &v45,
         v48) )
  {
    (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 104LL))(a1);
    sub_2626120(a1, a2);
    (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 112LL))(a1);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v48[0]);
  }
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _QWORD, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "WPDRes",
         0,
         0,
         &v41,
         &v43) )
  {
    v5 = (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) == 0;
    v6 = *(_QWORD *)a1;
    if ( v5 )
    {
      (*(void (__fastcall **)(__int64))(v6 + 104))(a1);
      v10 = (__int64 *)a1;
      (*(void (__fastcall **)(__int64 **, __int64))(*(_QWORD *)a1 + 136LL))(&v45, a1);
      v15 = v45;
      v40 = v46;
      if ( v45 != (__int64 *)v46 )
      {
        v37 = (_QWORD *)a2;
        do
        {
          v16 = (_BYTE *)*v15;
          v38 = v15[1];
          if ( sub_C93C90(*v15, v38, 0, (unsigned __int64 *)v48) )
          {
            v26 = *(void (__fastcall **)(__int64, __int64 *))(*(_QWORD *)a1 + 248LL);
            v48[0] = (__int64)"key not an integer";
            v10 = v48;
            v51 = 1;
            v50 = 3;
            v26(a1, v48);
          }
          else
          {
            v17 = v48[0];
            v18 = (_QWORD *)v37[7];
            v19 = v37 + 6;
            if ( !v18 )
              goto LABEL_28;
            do
            {
              while ( 1 )
              {
                v20 = v18[2];
                v21 = v18[3];
                if ( v48[0] <= v18[4] )
                  break;
                v18 = (_QWORD *)v18[3];
                if ( !v21 )
                  goto LABEL_26;
              }
              v19 = v18;
              v18 = (_QWORD *)v18[2];
            }
            while ( v20 );
LABEL_26:
            if ( v37 + 6 == v19 || v48[0] < v19[4] )
            {
LABEL_28:
              v31 = v37 + 6;
              v32 = v19;
              v22 = (unsigned __int64 *)sub_22077B0(0x80u);
              v22[4] = v17;
              memset(v22 + 5, 0, 0x58u);
              v34 = (__int64)v22;
              v22[6] = (unsigned __int64)(v22 + 8);
              v30 = v22 + 8;
              v22[13] = (unsigned __int64)(v22 + 11);
              v22[14] = (unsigned __int64)(v22 + 11);
              v23 = sub_9D7FB0(v37 + 5, v32, v22 + 4);
              if ( v24 )
              {
                v25 = v23 || v31 == v24 || v17 < v24[4];
                sub_220F040(v25, v34, v24, v31);
                v19 = (_QWORD *)v34;
                ++v37[10];
              }
              else
              {
                v33 = v23;
                sub_261C160(0);
                v27 = v34;
                v28 = v33;
                v29 = *(_QWORD *)(v34 + 48);
                if ( v30 != (unsigned __int64 *)v29 )
                {
                  j_j___libc_free_0(v29);
                  v28 = v33;
                  v27 = v34;
                }
                v36 = v28;
                j_j___libc_free_0(v27);
                v19 = v36;
              }
            }
            if ( v16 )
            {
              v35 = v19;
              v48[0] = (__int64)v49;
              sub_2619AF0(v48, v16, (__int64)&v16[v38]);
              v10 = (__int64 *)v48[0];
              v19 = v35;
            }
            else
            {
              v48[1] = 0;
              v48[0] = (__int64)v49;
              v10 = v49;
              LOBYTE(v49[0]) = 0;
            }
            v39 = v19;
            if ( (*(unsigned __int8 (__fastcall **)(__int64, __int64 *, __int64, _QWORD, char *, __int64 **))(*(_QWORD *)a1 + 120LL))(
                   a1,
                   v10,
                   1,
                   0,
                   &v42,
                   &v44) )
            {
              sub_262BF20(a1, (__int64)(v39 + 5));
              v10 = v44;
              (*(void (__fastcall **)(__int64, __int64 *))(*(_QWORD *)a1 + 128LL))(a1, v44);
            }
            if ( (_QWORD *)v48[0] != v49 )
            {
              v10 = (__int64 *)(v49[0] + 1LL);
              j_j___libc_free_0(v48[0]);
            }
          }
          v15 += 2;
        }
        while ( (__int64 *)v40 != v15 );
        v40 = (unsigned __int64)v45;
      }
      if ( v40 )
      {
        v10 = (__int64 *)(v47[0] - v40);
        j_j___libc_free_0(v40);
      }
    }
    else
    {
      (*(void (__fastcall **)(__int64))(v6 + 104))(a1);
      v8 = *(_QWORD *)(a2 + 64);
      v9 = (char *)v47 + 5;
      v10 = (__int64 *)&v44;
      if ( v8 != a2 + 48 )
      {
        v11 = *(_QWORD *)(a2 + 64);
        do
        {
          v13 = *(_QWORD *)(v11 + 32);
          if ( v13 )
          {
            v12 = (char *)v47 + 5;
            do
            {
              *--v12 = v13 % 0xA + 48;
              v14 = v13;
              v13 /= 0xAu;
            }
            while ( v14 > 9 );
          }
          else
          {
            BYTE4(v47[0]) = 48;
            v12 = (char *)v47 + 4;
          }
          v48[0] = (__int64)v49;
          sub_261A960(v48, v12, (__int64)v47 + 5);
          v10 = (__int64 *)v48[0];
          if ( (*(unsigned __int8 (__fastcall **)(__int64, __int64, __int64, _QWORD, __int64 **, __int64 **))(*(_QWORD *)a1 + 120LL))(
                 a1,
                 v48[0],
                 1,
                 0,
                 &v44,
                 &v45) )
          {
            sub_262BF20(a1, v11 + 40);
            v10 = v45;
            (*(void (__fastcall **)(__int64, __int64 *))(*(_QWORD *)a1 + 128LL))(a1, v45);
          }
          if ( (_QWORD *)v48[0] != v49 )
          {
            v10 = (__int64 *)(v49[0] + 1LL);
            j_j___libc_free_0(v48[0]);
          }
          v11 = sub_220EEE0(v11);
        }
        while ( a2 + 48 != v11 );
      }
    }
    (*(void (__fastcall **)(__int64, __int64 *, __int64, char *))(*(_QWORD *)a1 + 112LL))(a1, v10, v7, v9);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v43);
  }
  return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 112LL))(a1);
}
