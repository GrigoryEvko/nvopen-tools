// Function: sub_25E1B40
// Address: 0x25e1b40
//
__int64 __fastcall sub_25E1B40(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v3; // r12d
  __int64 v6; // r12
  __int64 v7; // rax
  __int64 v8; // rsi
  __int64 v9; // r8
  __int64 v10; // r9
  char *v11; // rax
  char *v12; // rsi
  __int64 v13; // rdx
  char *v14; // rcx
  char *v15; // rax
  __int64 *v16; // r14
  __int64 *v17; // rax
  __int64 v18; // rdi
  __int64 *v19; // rbx
  __int64 v20; // rsi
  _BYTE v21[8]; // [rsp+8h] [rbp-2D8h] BYREF
  int *v22; // [rsp+10h] [rbp-2D0h] BYREF
  __int64 *v23; // [rsp+18h] [rbp-2C8h]
  int v24; // [rsp+20h] [rbp-2C0h] BYREF
  unsigned int v25; // [rsp+28h] [rbp-2B8h]
  _QWORD v26[6]; // [rsp+30h] [rbp-2B0h] BYREF
  __int64 v27; // [rsp+60h] [rbp-280h]
  __int64 v28; // [rsp+68h] [rbp-278h]
  __int64 v29; // [rsp+70h] [rbp-270h]
  __int64 v30; // [rsp+78h] [rbp-268h]
  char *v31; // [rsp+80h] [rbp-260h]
  __int64 v32; // [rsp+88h] [rbp-258h]
  char v33; // [rsp+90h] [rbp-250h] BYREF
  __int64 v34; // [rsp+B0h] [rbp-230h]
  __int64 v35; // [rsp+B8h] [rbp-228h]
  __int64 v36; // [rsp+C0h] [rbp-220h]
  int v37; // [rsp+C8h] [rbp-218h]
  char *v38; // [rsp+D0h] [rbp-210h]
  __int64 v39; // [rsp+D8h] [rbp-208h]
  char v40; // [rsp+E0h] [rbp-200h] BYREF
  __int64 v41; // [rsp+1E0h] [rbp-100h]
  char *v42; // [rsp+1E8h] [rbp-F8h]
  __int64 v43; // [rsp+1F0h] [rbp-F0h]
  int v44; // [rsp+1F8h] [rbp-E8h]
  char v45; // [rsp+1FCh] [rbp-E4h]
  char v46; // [rsp+200h] [rbp-E0h] BYREF
  __int64 v47; // [rsp+240h] [rbp-A0h]
  char *v48; // [rsp+248h] [rbp-98h]
  __int64 v49; // [rsp+250h] [rbp-90h]
  int v50; // [rsp+258h] [rbp-88h]
  char v51; // [rsp+25Ch] [rbp-84h]
  char v52; // [rsp+260h] [rbp-80h] BYREF
  __int64 v53; // [rsp+2A0h] [rbp-40h]
  __int64 v54; // [rsp+2A8h] [rbp-38h]

  v3 = 0;
  if ( !sub_B2FC80(a1) )
  {
    v27 = 0;
    v26[1] = 8;
    v26[0] = sub_22077B0(0x40u);
    v6 = v26[0] + 24LL;
    v7 = sub_22077B0(0x200u);
    v26[5] = v26[0] + 24LL;
    v32 = 0x400000000LL;
    v26[4] = v7 + 512;
    v29 = v7 + 512;
    v31 = &v33;
    v38 = &v40;
    v42 = &v46;
    *(_QWORD *)(v26[0] + 24LL) = v7;
    v26[3] = v7;
    v30 = v6;
    v28 = v7;
    v26[2] = v7;
    v34 = 0;
    v35 = 0;
    v36 = 0;
    v37 = 0;
    v39 = 0x2000000000LL;
    v41 = 0;
    v43 = 8;
    v44 = 0;
    v45 = 1;
    v47 = 0;
    v48 = &v52;
    v49 = 8;
    v50 = 0;
    v51 = 1;
    v53 = a2;
    v54 = a3;
    if ( v7 )
    {
      *(_QWORD *)v7 = 0;
      *(_QWORD *)(v7 + 8) = 0;
      *(_QWORD *)(v7 + 16) = 0;
      *(_DWORD *)(v7 + 24) = 0;
    }
    v8 = a1;
    v27 = v7 + 32;
    v22 = &v24;
    v23 = 0;
    v3 = sub_29D2770(v26, a1, v21, &v22);
    if ( v22 != &v24 )
      _libc_free((unsigned __int64)v22);
    if ( (_BYTE)v3 )
    {
      sub_25E1800((__int64)&v22, (__int64)v26);
      v9 = (__int64)v23;
      v10 = 2LL * v25;
      if ( v24 )
      {
        v16 = &v23[v10];
        if ( v23 != &v23[v10] )
        {
          v17 = v23;
          while ( 1 )
          {
            v18 = *v17;
            v19 = v17;
            if ( *v17 != -4096 && v18 != -8192 )
              break;
            v17 += 2;
            if ( v16 == v17 )
              goto LABEL_10;
          }
          if ( v16 != v17 )
          {
            while ( 1 )
            {
              v20 = v19[1];
              v19 += 2;
              sub_B30160(v18, v20);
              if ( v19 == v16 )
                break;
              while ( *v19 == -8192 || *v19 == -4096 )
              {
                v19 += 2;
                if ( v16 == v19 )
                  goto LABEL_32;
              }
              if ( v16 == v19 )
                break;
              v18 = *v19;
            }
LABEL_32:
            v9 = (__int64)v23;
            v10 = 2LL * v25;
          }
        }
      }
LABEL_10:
      v11 = v42;
      if ( v45 )
        v12 = &v42[8 * HIDWORD(v43)];
      else
        v12 = &v42[8 * (unsigned int)v43];
      if ( v42 != v12 )
      {
        while ( 1 )
        {
          v13 = *(_QWORD *)v11;
          v14 = v11;
          if ( *(_QWORD *)v11 < 0xFFFFFFFFFFFFFFFELL )
            break;
          v11 += 8;
          if ( v12 == v11 )
            goto LABEL_20;
        }
        while ( v12 != v14 )
        {
          v15 = v14 + 8;
          *(_BYTE *)(v13 + 80) |= 1u;
          if ( v14 + 8 == v12 )
            break;
          while ( 1 )
          {
            v13 = *(_QWORD *)v15;
            v14 = v15;
            if ( *(_QWORD *)v15 < 0xFFFFFFFFFFFFFFFELL )
              break;
            v15 += 8;
            if ( v12 == v15 )
              goto LABEL_20;
          }
        }
      }
LABEL_20:
      v8 = v10 * 8;
      sub_C7D6A0(v9, v10 * 8, 8);
    }
    sub_25DE530((__int64)v26, v8);
  }
  return v3;
}
