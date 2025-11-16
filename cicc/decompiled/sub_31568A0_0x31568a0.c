// Function: sub_31568A0
// Address: 0x31568a0
//
__int64 __fastcall sub_31568A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rdx
  __int64 v6; // rcx
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // rax
  _QWORD *v9; // rdi
  __int64 v11; // rdx
  _QWORD *v12; // rax
  int v13; // ecx
  __int64 v14; // rax
  int v15; // edx
  __int64 v16; // rax
  int i; // [rsp+1Ch] [rbp-D4h]
  int v18; // [rsp+24h] [rbp-CCh] BYREF
  __int64 v19; // [rsp+28h] [rbp-C8h] BYREF
  void *v20[4]; // [rsp+30h] [rbp-C0h] BYREF
  char v21; // [rsp+50h] [rbp-A0h]
  char v22; // [rsp+51h] [rbp-9Fh]
  __int64 v23; // [rsp+60h] [rbp-90h] BYREF
  int v24; // [rsp+68h] [rbp-88h] BYREF
  _QWORD *v25; // [rsp+70h] [rbp-80h]
  int *v26; // [rsp+78h] [rbp-78h]
  int *v27; // [rsp+80h] [rbp-70h]
  __int64 v28; // [rsp+88h] [rbp-68h]
  __int64 v29; // [rsp+90h] [rbp-60h] BYREF
  int v30; // [rsp+98h] [rbp-58h] BYREF
  _QWORD *v31; // [rsp+A0h] [rbp-50h]
  int *v32; // [rsp+A8h] [rbp-48h]
  int *v33; // [rsp+B0h] [rbp-40h]
  __int64 v34; // [rsp+B8h] [rbp-38h]

  sub_3154C90(&v23, a2, a3, a4);
  v7 = v23 & 0xFFFFFFFFFFFFFFFELL;
  if ( (v23 & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    *(_BYTE *)(a1 + 96) |= 3u;
    *(_QWORD *)a1 = v7;
  }
  else
  {
    v24 = 0;
    v25 = 0;
    v26 = &v24;
    v27 = &v24;
    v28 = 0;
    v30 = 0;
    v31 = 0;
    v32 = &v30;
    v33 = &v30;
    v34 = 0;
    v18 = 0;
    for ( i = 2; (unsigned __int8)sub_3154990(a2, &v18, v5, v6); i = 1 )
    {
      if ( v18 == 9 )
      {
        sub_3156040((__int64 *)v20, a2, &v23);
        v8 = (unsigned __int64)v20[0] & 0xFFFFFFFFFFFFFFFELL;
        if ( ((unsigned __int64)v20[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
        {
LABEL_7:
          *(_BYTE *)(a1 + 96) |= 3u;
          v9 = v31;
          *(_QWORD *)a1 = v8;
          goto LABEL_8;
        }
      }
      else
      {
        if ( v18 != 12 )
        {
          v22 = 1;
          v20[0] = "Unexpected section";
          v21 = 3;
          sub_31542E0(&v19, a2, v20);
          v16 = v19;
          *(_BYTE *)(a1 + 96) |= 3u;
          v9 = v31;
          *(_QWORD *)a1 = v16 & 0xFFFFFFFFFFFFFFFELL;
          goto LABEL_8;
        }
        sub_3156450((__int64 *)v20, a2, &v29);
        v8 = (unsigned __int64)v20[0] & 0xFFFFFFFFFFFFFFFELL;
        if ( ((unsigned __int64)v20[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
          goto LABEL_7;
      }
      if ( i == 1 )
        break;
    }
    v11 = a1 + 8;
    *(_BYTE *)(a1 + 96) = *(_BYTE *)(a1 + 96) & 0xFC | 2;
    v12 = v25;
    if ( v25 )
    {
      v13 = v24;
      *(_QWORD *)(a1 + 16) = v25;
      *(_DWORD *)(a1 + 8) = v13;
      *(_QWORD *)(a1 + 24) = v26;
      *(_QWORD *)(a1 + 32) = v27;
      v12[1] = v11;
      v25 = 0;
      *(_QWORD *)(a1 + 40) = v28;
      v26 = &v24;
      v27 = &v24;
      v28 = 0;
    }
    else
    {
      *(_DWORD *)(a1 + 8) = 0;
      *(_QWORD *)(a1 + 16) = 0;
      *(_QWORD *)(a1 + 24) = v11;
      *(_QWORD *)(a1 + 32) = v11;
      *(_QWORD *)(a1 + 40) = 0;
    }
    v9 = v31;
    v14 = a1 + 56;
    if ( v31 )
    {
      v15 = v30;
      *(_QWORD *)(a1 + 64) = v31;
      *(_DWORD *)(a1 + 56) = v15;
      *(_QWORD *)(a1 + 72) = v32;
      *(_QWORD *)(a1 + 80) = v33;
      v9[1] = v14;
      v9 = 0;
      v31 = 0;
      *(_QWORD *)(a1 + 88) = v34;
      v32 = &v30;
      v33 = &v30;
      v34 = 0;
    }
    else
    {
      *(_DWORD *)(a1 + 56) = 0;
      *(_QWORD *)(a1 + 64) = 0;
      *(_QWORD *)(a1 + 72) = v14;
      *(_QWORD *)(a1 + 80) = v14;
      *(_QWORD *)(a1 + 88) = 0;
    }
LABEL_8:
    sub_3153A90(v9);
    sub_3153D30(v25);
  }
  return a1;
}
