// Function: sub_30DEDC0
// Address: 0x30dedc0
//
__int64 __fastcall sub_30DEDC0(
        __int64 a1,
        unsigned __int8 *a2,
        __int64 a3,
        int *a4,
        __int64 *a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9,
        __int64 a10,
        __int64 a11,
        __int64 a12,
        __int64 a13)
{
  char *v15; // rax
  char v16; // dl
  void *v18; // rax
  int v19; // eax
  const char *v20; // rax
  unsigned int v21; // eax
  unsigned int v22; // eax
  bool v23; // zf
  const void *v26; // [rsp+20h] [rbp-3E0h] BYREF
  unsigned int v27; // [rsp+28h] [rbp-3D8h]
  const void *v28; // [rsp+30h] [rbp-3D0h] BYREF
  unsigned int v29; // [rsp+38h] [rbp-3C8h]
  char v30; // [rsp+40h] [rbp-3C0h]
  const void *v31; // [rsp+50h] [rbp-3B0h] BYREF
  unsigned int v32; // [rsp+58h] [rbp-3A8h]
  const void *v33; // [rsp+60h] [rbp-3A0h] BYREF
  unsigned int v34; // [rsp+68h] [rbp-398h]
  char v35; // [rsp+70h] [rbp-390h]
  _QWORD v36[9]; // [rsp+80h] [rbp-380h] BYREF
  __int64 v37; // [rsp+C8h] [rbp-338h]
  __int64 v38; // [rsp+340h] [rbp-C0h]
  int v39; // [rsp+34Ch] [rbp-B4h]
  char v40; // [rsp+358h] [rbp-A8h]
  char v41; // [rsp+359h] [rbp-A7h]
  const void *v42; // [rsp+360h] [rbp-A0h] BYREF
  unsigned int v43; // [rsp+368h] [rbp-98h]
  const void *v44; // [rsp+370h] [rbp-90h] BYREF
  unsigned int v45; // [rsp+378h] [rbp-88h]
  char v46; // [rsp+380h] [rbp-80h]

  v15 = sub_30D6600(a2, a3, a5, a9, a10);
  if ( !v16 )
  {
    sub_30D4900((__int64)v36, a3, (__int64)a2, a4, (__int64)a5, a6, a7, a8, a11, a12, a9, a10, a13, 1, 0);
    v18 = (void *)sub_30D5340((__int64)v36);
    if ( v18 )
    {
      if ( v41 )
      {
LABEL_9:
        v30 = 0;
        if ( !v46 )
        {
          *(_QWORD *)a1 = 0x7FFFFFFF;
          *(_DWORD *)(a1 + 8) = 0;
          *(_QWORD *)(a1 + 16) = "cost over benefit";
          *(_BYTE *)(a1 + 56) = 0;
          goto LABEL_11;
        }
        v27 = v43;
        if ( v43 > 0x40 )
          sub_C43780((__int64)&v26, &v42);
        else
          v26 = v42;
        v29 = v45;
        if ( v45 > 0x40 )
          sub_C43780((__int64)&v28, &v44);
        else
          v28 = v44;
        v30 = 1;
        v35 = 0;
        v32 = v27;
        if ( v27 > 0x40 )
          sub_C43780((__int64)&v31, &v26);
        else
          v31 = v26;
        v34 = v29;
        if ( v29 > 0x40 )
          sub_C43780((__int64)&v33, &v28);
        else
          v33 = v28;
        v35 = 1;
        v20 = "cost over benefit";
        *(_QWORD *)a1 = 0x7FFFFFFF;
        *(_DWORD *)(a1 + 8) = 0;
        goto LABEL_36;
      }
      if ( !v40 )
        goto LABEL_14;
LABEL_23:
      v19 = v39;
      *(_QWORD *)(a1 + 16) = 0;
      *(_BYTE *)(a1 + 56) = 0;
      *(_DWORD *)a1 = v19;
      *(_QWORD *)(a1 + 4) = v38;
      goto LABEL_15;
    }
    if ( v37 + 72 == (*(_QWORD *)(v37 + 72) & 0xFFFFFFFFFFFFFFF8LL) )
    {
      if ( v41 )
        goto LABEL_19;
      if ( v40 )
        goto LABEL_23;
    }
    else
    {
      v18 = sub_30DC7E0(v36);
      if ( v41 )
      {
        if ( v18 )
          goto LABEL_9;
LABEL_19:
        v30 = 0;
        if ( !v46 )
        {
          *(_DWORD *)(a1 + 8) = 0;
          *(_QWORD *)a1 = 0x80000000LL;
          *(_QWORD *)(a1 + 16) = "benefit over cost";
          *(_BYTE *)(a1 + 56) = 0;
LABEL_11:
          if ( v30 )
            sub_26C3CC0((__int64)&v26);
          goto LABEL_15;
        }
        v27 = v43;
        if ( v43 > 0x40 )
          sub_C43780((__int64)&v26, &v42);
        else
          v26 = v42;
        v29 = v45;
        if ( v45 > 0x40 )
          sub_C43780((__int64)&v28, &v44);
        else
          v28 = v44;
        v30 = 1;
        v35 = 0;
        v32 = v27;
        if ( v27 > 0x40 )
          sub_C43780((__int64)&v31, &v26);
        else
          v31 = v26;
        v34 = v29;
        if ( v29 > 0x40 )
          sub_C43780((__int64)&v33, &v28);
        else
          v33 = v28;
        v35 = 1;
        *(_QWORD *)a1 = 0x80000000LL;
        v20 = "benefit over cost";
        *(_DWORD *)(a1 + 8) = 0;
LABEL_36:
        *(_QWORD *)(a1 + 16) = v20;
        v21 = v32;
        *(_BYTE *)(a1 + 56) = 0;
        *(_DWORD *)(a1 + 32) = v21;
        if ( v21 > 0x40 )
          sub_C43780(a1 + 24, &v31);
        else
          *(_QWORD *)(a1 + 24) = v31;
        v22 = v34;
        *(_DWORD *)(a1 + 48) = v34;
        if ( v22 > 0x40 )
          sub_C43780(a1 + 40, &v33);
        else
          *(_QWORD *)(a1 + 40) = v33;
        v23 = v35 == 0;
        *(_BYTE *)(a1 + 56) = 1;
        if ( !v23 )
        {
          v35 = 0;
          sub_969240((__int64 *)&v33);
          sub_969240((__int64 *)&v31);
        }
        goto LABEL_11;
      }
      if ( v40 )
        goto LABEL_23;
      if ( v18 )
      {
LABEL_14:
        *(_QWORD *)a1 = 0x7FFFFFFF;
        *(_DWORD *)(a1 + 8) = 0;
        *(_QWORD *)(a1 + 16) = v18;
        *(_BYTE *)(a1 + 56) = 0;
LABEL_15:
        sub_30D30A0((__int64)v36);
        return a1;
      }
    }
    *(_DWORD *)(a1 + 8) = 0;
    *(_QWORD *)a1 = 0x80000000LL;
    *(_QWORD *)(a1 + 16) = "empty function";
    *(_BYTE *)(a1 + 56) = 0;
    goto LABEL_15;
  }
  if ( v15 )
  {
    *(_QWORD *)a1 = 0x7FFFFFFF;
    *(_DWORD *)(a1 + 8) = 0;
    *(_QWORD *)(a1 + 16) = v15;
  }
  else
  {
    *(_DWORD *)(a1 + 8) = 0;
    *(_QWORD *)a1 = 0x80000000LL;
    *(_QWORD *)(a1 + 16) = "always inline attribute";
  }
  *(_BYTE *)(a1 + 56) = 0;
  return a1;
}
