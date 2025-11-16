// Function: sub_14558A0
// Address: 0x14558a0
//
__int64 __fastcall sub_14558A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5, bool a6)
{
  unsigned int v8; // r15d
  unsigned int v9; // r15d
  unsigned int v10; // eax
  unsigned int v11; // eax
  int v13; // eax
  int v14; // eax
  unsigned int v15; // eax
  unsigned int v16; // eax
  __int64 *v17; // rax
  int v18; // edx
  __int64 v19; // r8
  unsigned int v20; // edx
  __int64 v21; // rax
  __int64 v22; // rcx
  unsigned int v23; // eax
  unsigned int v24; // eax
  int v25; // [rsp+10h] [rbp-D0h]
  bool v26; // [rsp+18h] [rbp-C8h]
  bool v27; // [rsp+18h] [rbp-C8h]
  bool v28; // [rsp+18h] [rbp-C8h]
  bool v29; // [rsp+20h] [rbp-C0h]
  unsigned int v30; // [rsp+28h] [rbp-B8h]
  unsigned __int64 v32; // [rsp+30h] [rbp-B0h] BYREF
  unsigned int v33; // [rsp+38h] [rbp-A8h]
  unsigned __int64 v34; // [rsp+40h] [rbp-A0h] BYREF
  unsigned int v35; // [rsp+48h] [rbp-98h]
  __int64 v36; // [rsp+50h] [rbp-90h] BYREF
  int v37; // [rsp+58h] [rbp-88h]
  unsigned __int64 v38; // [rsp+60h] [rbp-80h] BYREF
  unsigned int v39; // [rsp+68h] [rbp-78h]
  unsigned __int64 v40; // [rsp+70h] [rbp-70h] BYREF
  unsigned int v41; // [rsp+78h] [rbp-68h]
  __int64 v42; // [rsp+80h] [rbp-60h] BYREF
  int v43; // [rsp+88h] [rbp-58h]
  __int64 v44; // [rsp+90h] [rbp-50h] BYREF
  unsigned int v45; // [rsp+98h] [rbp-48h]
  __int64 v46; // [rsp+A0h] [rbp-40h] BYREF
  int v47; // [rsp+A8h] [rbp-38h]

  v8 = *(_DWORD *)(a2 + 8);
  v29 = a6;
  if ( v8 > 0x40 )
  {
    v26 = a6;
    v13 = sub_16A57B0(a2);
    a6 = v26;
    if ( v8 - v13 <= 0x40 && !**(_QWORD **)a2 )
      goto LABEL_5;
  }
  else if ( !*(_QWORD *)a2 )
  {
    goto LABEL_5;
  }
  v9 = *(_DWORD *)(a4 + 8);
  if ( v9 <= 0x40 )
  {
    if ( !*(_QWORD *)a4 )
      goto LABEL_5;
LABEL_15:
    v28 = a6;
    if ( (unsigned __int8)sub_158A0B0(a3) )
      goto LABEL_30;
    if ( v28 )
    {
      v29 = sub_13D0200((__int64 *)a2, *(_DWORD *)(a2 + 8) - 1);
      sub_13A3E40((__int64)&v46, a2);
      sub_14536D0((__int64 *)a2, &v46);
      sub_135E100(&v46);
    }
    sub_135E0D0((__int64)&v44, *(_DWORD *)(a3 + 8), -1, 1u);
    sub_16A9D70(&v46, &v44, a2);
    v25 = sub_16A9900(&v46, a4);
    sub_135E100(&v46);
    sub_135E100(&v44);
    if ( v25 < 0 )
    {
LABEL_30:
      sub_15897D0(a1, a5, 1);
      return a1;
    }
    sub_16A7B50(&v32, a2, a4);
    sub_13A38D0((__int64)&v34, a3);
    sub_13A38D0((__int64)&v46, a3 + 16);
    sub_16A7800(&v46, 1);
    v37 = v47;
    v36 = v46;
    if ( v29 )
    {
      if ( v33 > 0x40 )
        sub_16A8F40(&v32);
      else
        v32 = ~v32 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v33);
      sub_16A7400(&v32);
      sub_16A7200(&v32, &v34);
      v23 = v33;
      v33 = 0;
      v39 = v23;
      v38 = v32;
      if ( !(unsigned __int8)sub_158B950(a3, &v38) )
      {
        v24 = v39;
        v39 = 0;
        v41 = v24;
        v40 = v38;
        v17 = &v36;
LABEL_22:
        v18 = *((_DWORD *)v17 + 2);
        *((_DWORD *)v17 + 2) = 0;
        v43 = v18;
        v42 = *v17;
        sub_16A7490(&v42, 1);
        v20 = v41;
        if ( v41 <= 0x40 )
        {
          v21 = v40;
          v22 = v42;
          if ( v40 != v42 )
            goto LABEL_25;
        }
        else
        {
          v30 = v41;
          if ( !(unsigned __int8)sub_16A5220(&v40, &v42) )
          {
            v21 = v40;
            v22 = v42;
            v20 = v30;
LABEL_25:
            v45 = v20;
            v46 = v22;
            v47 = v43;
            v44 = v21;
            v43 = 0;
            v41 = 0;
            sub_15898E0(a1, &v44, &v46, v22, v19);
            sub_135E100(&v44);
            sub_135E100(&v46);
LABEL_26:
            sub_135E100(&v42);
            sub_135E100((__int64 *)&v40);
LABEL_27:
            sub_135E100((__int64 *)&v38);
            sub_135E100(&v36);
            sub_135E100((__int64 *)&v34);
            sub_135E100((__int64 *)&v32);
            return a1;
          }
        }
        sub_15897D0(a1, a5, 1);
        goto LABEL_26;
      }
    }
    else
    {
      sub_16A7200(&v32, &v36);
      v15 = v33;
      v33 = 0;
      v39 = v15;
      v38 = v32;
      if ( !(unsigned __int8)sub_158B950(a3, &v38) )
      {
        v16 = v35;
        v35 = 0;
        v41 = v16;
        v40 = v34;
        v17 = (__int64 *)&v38;
        goto LABEL_22;
      }
    }
    sub_15897D0(a1, a5, 1);
    goto LABEL_27;
  }
  v27 = a6;
  v14 = sub_16A57B0(a4);
  a6 = v27;
  if ( v9 - v14 > 0x40 || **(_QWORD **)a4 )
    goto LABEL_15;
LABEL_5:
  v10 = *(_DWORD *)(a3 + 8);
  *(_DWORD *)(a1 + 8) = v10;
  if ( v10 > 0x40 )
    sub_16A4FD0(a1, a3);
  else
    *(_QWORD *)a1 = *(_QWORD *)a3;
  v11 = *(_DWORD *)(a3 + 24);
  *(_DWORD *)(a1 + 24) = v11;
  if ( v11 > 0x40 )
    sub_16A4FD0(a1 + 16, a3 + 16);
  else
    *(_QWORD *)(a1 + 16) = *(_QWORD *)(a3 + 16);
  return a1;
}
