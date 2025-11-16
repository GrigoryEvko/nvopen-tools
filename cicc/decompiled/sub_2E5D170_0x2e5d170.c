// Function: sub_2E5D170
// Address: 0x2e5d170
//
__int64 __fastcall sub_2E5D170(__int64 a1, __int64 a2, __int64 *a3, __int64 a4)
{
  __int64 v6; // r13
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r9
  __int64 **v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  void **v27; // rax
  __int64 **v28; // rsi
  __int64 v29; // [rsp+8h] [rbp-3D8h]
  __int64 v30; // [rsp+10h] [rbp-3D0h] BYREF
  void **v31; // [rsp+18h] [rbp-3C8h]
  unsigned int v32; // [rsp+20h] [rbp-3C0h]
  unsigned int v33; // [rsp+24h] [rbp-3BCh]
  char v34; // [rsp+2Ch] [rbp-3B4h]
  _BYTE v35[16]; // [rsp+30h] [rbp-3B0h] BYREF
  _BYTE v36[8]; // [rsp+40h] [rbp-3A0h] BYREF
  unsigned __int64 v37; // [rsp+48h] [rbp-398h]
  int v38; // [rsp+54h] [rbp-38Ch]
  int v39; // [rsp+58h] [rbp-388h]
  char v40; // [rsp+5Ch] [rbp-384h]
  _BYTE v41[16]; // [rsp+60h] [rbp-380h] BYREF
  _QWORD v42[5]; // [rsp+70h] [rbp-370h] BYREF
  __int16 v43; // [rsp+98h] [rbp-348h]
  __int64 v44; // [rsp+A0h] [rbp-340h]
  __int64 v45; // [rsp+A8h] [rbp-338h]
  __int64 v46; // [rsp+B0h] [rbp-330h]
  int v47; // [rsp+B8h] [rbp-328h]
  __int64 v48; // [rsp+C0h] [rbp-320h]
  __int64 v49; // [rsp+C8h] [rbp-318h]
  __int64 v50; // [rsp+D0h] [rbp-310h]
  int v51; // [rsp+D8h] [rbp-308h]
  __int64 v52; // [rsp+E0h] [rbp-300h]
  __int64 v53; // [rsp+E8h] [rbp-2F8h]
  __int64 v54; // [rsp+F0h] [rbp-2F0h]
  int v55; // [rsp+F8h] [rbp-2E8h]
  __int64 v56; // [rsp+100h] [rbp-2E0h]
  __int64 v57; // [rsp+108h] [rbp-2D8h]
  __int64 v58; // [rsp+110h] [rbp-2D0h]
  char *v59; // [rsp+118h] [rbp-2C8h]
  __int64 v60; // [rsp+120h] [rbp-2C0h]
  char v61; // [rsp+128h] [rbp-2B8h] BYREF
  _QWORD *v62; // [rsp+148h] [rbp-298h]
  __int64 v63; // [rsp+150h] [rbp-290h]
  _QWORD v64[5]; // [rsp+158h] [rbp-288h] BYREF
  int v65; // [rsp+180h] [rbp-260h]
  __int64 v66; // [rsp+188h] [rbp-258h]
  char *v67; // [rsp+190h] [rbp-250h]
  __int64 v68; // [rsp+198h] [rbp-248h]
  char v69; // [rsp+1A0h] [rbp-240h] BYREF
  int v70; // [rsp+3A0h] [rbp-40h]

  v6 = sub_2EB2140(a4, &unk_501FE48);
  v29 = sub_2EB2140(a4, &unk_501EC10);
  v7 = sub_2EB2140(a4, &unk_502D268);
  v8 = sub_2EB2140(a4, &unk_50209D0);
  v9 = sub_BC1CD0(*(_QWORD *)(v8 + 8), &unk_4F89C30, *a3);
  v42[2] = v6 + 8;
  v45 = v9 + 8;
  v59 = &v61;
  v60 = 0x400000000LL;
  v42[4] = v29 + 8;
  v62 = v64;
  v43 = 257;
  v44 = v7 + 8;
  v42[0] = 0;
  v42[1] = 0;
  v42[3] = 0;
  v46 = 0;
  v47 = 0;
  v48 = 0;
  v49 = 0;
  v50 = 0;
  v51 = 0;
  v52 = 0;
  v53 = 0;
  v54 = 0;
  v55 = 0;
  v56 = 0;
  v57 = 0;
  v58 = 0;
  v63 = 0;
  v64[0] = 0;
  v64[1] = 1;
  memset(&v64[2], 0, 24);
  v65 = 0;
  v66 = 0;
  v67 = &v69;
  v68 = 0x4000000000LL;
  v70 = 0;
  if ( !(unsigned __int8)sub_2E5C7D0((__int64)v42, (__int64)a3) )
  {
    *(_QWORD *)(a1 + 8) = a1 + 32;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 56) = a1 + 80;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
    goto LABEL_3;
  }
  sub_2EAFFB0(&v30);
  sub_2E4FDC0((__int64)&v30, (__int64)&unk_50208B0, v11, v12, v13, v14);
  sub_2E4FDC0((__int64)&v30, (__int64)&unk_501FE48, v15, v16, v17, v18);
  sub_2E4FDC0((__int64)&v30, (__int64)&unk_501EC10, v19, v20, v21, v22);
  if ( v38 == v39 )
  {
    if ( v34 )
    {
      v27 = v31;
      v28 = (__int64 **)&v31[v33];
      v24 = v33;
      v23 = (__int64 **)v31;
      if ( v31 != (void **)v28 )
      {
        while ( *v23 != &qword_4F82400 )
        {
          if ( v28 == ++v23 )
          {
LABEL_9:
            while ( *v27 != &unk_4F82408 )
            {
              if ( v23 == (__int64 **)++v27 )
                goto LABEL_14;
            }
            goto LABEL_10;
          }
        }
        goto LABEL_10;
      }
      goto LABEL_14;
    }
    if ( sub_C8CA60((__int64)&v30, (__int64)&qword_4F82400) )
      goto LABEL_10;
  }
  if ( !v34 )
  {
LABEL_16:
    sub_C8CC70((__int64)&v30, (__int64)&unk_4F82408, (__int64)v23, v24, v25, v26);
    goto LABEL_10;
  }
  v27 = v31;
  v24 = v33;
  v23 = (__int64 **)&v31[v33];
  if ( v31 != (void **)v23 )
    goto LABEL_9;
LABEL_14:
  if ( (unsigned int)v24 >= v32 )
    goto LABEL_16;
  v33 = v24 + 1;
  *v23 = (__int64 *)&unk_4F82408;
  ++v30;
LABEL_10:
  sub_C8CF70(a1, (void *)(a1 + 32), 2, (__int64)v35, (__int64)&v30);
  sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v41, (__int64)v36);
  if ( !v40 )
    _libc_free(v37);
  if ( !v34 )
    _libc_free((unsigned __int64)v31);
LABEL_3:
  sub_2E50030((__int64)v42);
  return a1;
}
