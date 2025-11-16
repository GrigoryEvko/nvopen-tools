// Function: sub_2D50FB0
// Address: 0x2d50fb0
//
__int64 __fastcall sub_2D50FB0(__int64 a1, __int64 *a2, char *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rdx
  __int64 v7; // r9
  __int64 v8; // rax
  __int64 v10; // r8
  __int64 v11; // r9
  int v12; // edx
  char v13; // al
  const char *v14; // rcx
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 *v17; // rax
  __int64 v18; // rax
  const char *v19; // rdx
  __int64 v20; // rax
  __int128 v21; // [rsp-30h] [rbp-130h]
  __int128 v22; // [rsp-30h] [rbp-130h]
  __int128 v23; // [rsp-20h] [rbp-120h]
  __int128 v24; // [rsp-20h] [rbp-120h]
  __int64 v25; // [rsp-10h] [rbp-110h]
  char *v26; // [rsp+0h] [rbp-100h] BYREF
  __int64 v27; // [rsp+8h] [rbp-F8h]
  unsigned __int64 v28; // [rsp+10h] [rbp-F0h] BYREF
  unsigned __int64 v29; // [rsp+18h] [rbp-E8h] BYREF
  __int64 v30[4]; // [rsp+20h] [rbp-E0h] BYREF
  __int64 v31; // [rsp+40h] [rbp-C0h]
  __int64 v32[2]; // [rsp+50h] [rbp-B0h] BYREF
  const char *v33; // [rsp+60h] [rbp-A0h]
  __int64 v34; // [rsp+68h] [rbp-98h]
  __int64 v35; // [rsp+70h] [rbp-90h]
  _QWORD v36[2]; // [rsp+80h] [rbp-80h] BYREF
  const char *v37; // [rsp+90h] [rbp-70h]
  __int64 v38; // [rsp+98h] [rbp-68h]
  __int64 v39; // [rsp+A0h] [rbp-60h]
  __int64 *v40; // [rsp+B0h] [rbp-50h] BYREF
  __int64 v41; // [rsp+B8h] [rbp-48h]
  _BYTE v42[64]; // [rsp+C0h] [rbp-40h] BYREF

  v26 = a3;
  v27 = a4;
  v40 = (__int64 *)v42;
  v41 = 0x200000000LL;
  sub_C93960(&v26, (__int64)&v40, 46, -1, 1, a6);
  if ( (unsigned int)v41 > 2 )
  {
    v36[0] = "unable to parse basic block id: '";
    v30[2] = (__int64)"'";
    v37 = v26;
    LOWORD(v31) = 770;
    v38 = v27;
    *((_QWORD *)&v23 + 1) = v30[3];
    *(_QWORD *)&v23 = "'";
    *((_QWORD *)&v21 + 1) = v30[1];
    *(_QWORD *)&v21 = v36;
    LOWORD(v39) = 1283;
    v30[0] = (__int64)v36;
    sub_2D507F0(v32, a2, v6, (__int64)"'", 770, v7, v21, v23, v31);
    v8 = v32[0];
    *(_BYTE *)(a1 + 8) |= 3u;
    *(_QWORD *)a1 = v8 & 0xFFFFFFFFFFFFFFFELL;
    goto LABEL_3;
  }
  if ( sub_C93C90(*v40, v40[1], 0xAu, &v28) )
  {
    LOWORD(v39) = 1283;
    v14 = "': unsigned integer expected";
    v36[0] = "unable to parse BB id: '";
    v15 = *v40;
    v37 = (const char *)*v40;
    v16 = v40[1];
    v33 = "': unsigned integer expected";
    v38 = v16;
    v17 = v36;
    LOWORD(v35) = 770;
    v32[0] = (__int64)v36;
    v25 = v35;
    *((_QWORD *)&v24 + 1) = v34;
    *(_QWORD *)&v24 = "': unsigned integer expected";
    *((_QWORD *)&v22 + 1) = v32[1];
  }
  else
  {
    v29 = 0;
    v12 = 0;
    if ( (unsigned int)v41 <= 1 )
    {
LABEL_8:
      v13 = *(_BYTE *)(a1 + 8);
      *(_DWORD *)(a1 + 4) = v12;
      *(_BYTE *)(a1 + 8) = v13 & 0xFC | 2;
      *(_DWORD *)a1 = v28;
      goto LABEL_3;
    }
    if ( !sub_C93C90(v40[2], v40[3], 0xAu, &v29) )
    {
      v12 = v29;
      goto LABEL_8;
    }
    v14 = "unable to parse clone id: '";
    v19 = (const char *)v40[2];
    v20 = v40[3];
    v32[0] = (__int64)"unable to parse clone id: '";
    v33 = v19;
    v34 = v20;
    v37 = "': unsigned integer expected";
    v15 = 770;
    LOWORD(v35) = 1283;
    v17 = v32;
    LOWORD(v39) = 770;
    v36[0] = v32;
    v25 = v39;
    *((_QWORD *)&v24 + 1) = v38;
    *(_QWORD *)&v24 = "': unsigned integer expected";
    *((_QWORD *)&v22 + 1) = v36[1];
  }
  *(_QWORD *)&v22 = v17;
  sub_2D507F0(v30, a2, v15, (__int64)v14, v10, v11, v22, v24, v25);
  v18 = v30[0];
  *(_BYTE *)(a1 + 8) |= 3u;
  *(_QWORD *)a1 = v18 & 0xFFFFFFFFFFFFFFFELL;
LABEL_3:
  if ( v40 != (__int64 *)v42 )
    _libc_free((unsigned __int64)v40);
  return a1;
}
