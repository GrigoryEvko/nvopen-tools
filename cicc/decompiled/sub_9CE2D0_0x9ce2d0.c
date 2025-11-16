// Function: sub_9CE2D0
// Address: 0x9ce2d0
//
__int64 __fastcall sub_9CE2D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r15
  unsigned int v6; // r13d
  __int64 v7; // rdx
  __int64 v8; // rdi
  __int64 v9; // r8
  __int64 v10; // rcx
  __int64 v11; // rax
  int v12; // ebx
  bool v13; // zf
  char v14; // al
  char v15; // di
  __int64 v16; // r13
  __int64 v17; // rax
  __int64 v19; // rax
  char v20; // al
  char v21; // al
  __int64 v22; // rax
  char v23; // [rsp+7h] [rbp-89h]
  char v24; // [rsp+7h] [rbp-89h]
  int v25; // [rsp+10h] [rbp-80h]
  int v26; // [rsp+14h] [rbp-7Ch]
  unsigned int v27; // [rsp+18h] [rbp-78h]
  unsigned int v28; // [rsp+18h] [rbp-78h]
  unsigned int i; // [rsp+1Ch] [rbp-74h]
  __int64 v30; // [rsp+20h] [rbp-70h]
  char v31; // [rsp+28h] [rbp-68h]
  char v32; // [rsp+28h] [rbp-68h]
  char v33; // [rsp+28h] [rbp-68h]
  __int64 v34; // [rsp+30h] [rbp-60h] BYREF
  char v35; // [rsp+38h] [rbp-58h]
  __int64 v36; // [rsp+40h] [rbp-50h] BYREF
  char v37; // [rsp+48h] [rbp-48h]
  _QWORD v38[8]; // [rsp+50h] [rbp-40h] BYREF

  v4 = a2;
  v6 = a3;
  sub_9C66D0((__int64)&v36, a2, a3, a4);
  v8 = v37 & 1;
  v37 &= ~2u;
  if ( (_BYTE)v8 )
  {
    v19 = v36;
    *(_BYTE *)(a1 + 8) |= 3u;
    v30 = 0;
    *(_QWORD *)a1 = v19;
LABEL_20:
    if ( v30 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v30 + 8LL))(v30);
    return a1;
  }
  v9 = v6 - 1;
  v32 = v31 & 0xFC;
  v10 = v9;
  v11 = 1LL << ((unsigned __int8)v6 - 1);
  LODWORD(v30) = v36;
  v26 = v11;
  if ( ((unsigned int)v11 & (unsigned int)v36) == 0 )
  {
    v20 = *(_BYTE *)(a1 + 8);
    *(_DWORD *)a1 = v36;
    *(_BYTE *)(a1 + 8) = v20 & 0xFC | 2;
    return a1;
  }
  v25 = v11 - 1;
  v12 = (v11 - 1) & v36;
  if ( (unsigned int)v9 <= 0x1F )
  {
    for ( i = v9; i <= 0x1F; i += v9 )
    {
      a2 = v4;
      v27 = v9;
      sub_9C66D0((__int64)&v34, v4, v6, v10);
      v9 = v27;
      v13 = (v35 & 1) == 0;
      v14 = v35 & 1;
      v35 &= ~2u;
      if ( v13 )
      {
        LODWORD(v36) = v34;
        v15 = v32 & 1;
        if ( (v32 & 1) == 0 )
          goto LABEL_7;
      }
      else
      {
        v36 = v34;
        v34 = 0;
        v15 = v32 & 1;
        if ( (v32 & 1) == 0 )
          goto LABEL_7;
      }
      v15 = 0;
      if ( v30 )
      {
        v23 = v14;
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v30 + 8LL))(v30);
        v14 = v23;
        v9 = v27;
        v15 = (v35 & 2) != 0;
      }
LABEL_7:
      v33 = v14 | v32 & 0xFE | 2;
      if ( v14 )
        v30 = v36;
      else
        LODWORD(v30) = v36;
      if ( v15 )
        sub_9CDF70(&v34);
      if ( (v35 & 1) != 0 && v34 )
      {
        v24 = v14;
        v28 = v9;
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v34 + 8LL))(v34);
        v14 = v24;
        v9 = v28;
      }
      v32 = (2 * v14) | v33 & 0xFD;
      if ( v14 )
      {
        v22 = v30;
        *(_BYTE *)(a1 + 8) |= 3u;
        v30 = 0;
        *(_QWORD *)a1 = v22;
        goto LABEL_20;
      }
      v8 = (unsigned int)v30;
      v10 = (unsigned __int8)i;
      v12 |= ((unsigned int)v30 & v25) << i;
      if ( ((unsigned int)v30 & v26) == 0 )
      {
        v21 = *(_BYTE *)(a1 + 8);
        *(_DWORD *)a1 = v12;
        *(_BYTE *)(a1 + 8) = v21 & 0xFC | 2;
        goto LABEL_17;
      }
    }
  }
  v16 = sub_2241E50(v8, a2, v7, v10, v9);
  v36 = (__int64)v38;
  sub_9C2D70(&v36, "Unterminated VBR", (__int64)"");
  sub_C63F00(&v34, &v36, 84, v16);
  if ( (_QWORD *)v36 != v38 )
    j_j___libc_free_0(v36, v38[0] + 1LL);
  v17 = v34;
  *(_BYTE *)(a1 + 8) |= 3u;
  *(_QWORD *)a1 = v17 & 0xFFFFFFFFFFFFFFFELL;
LABEL_17:
  if ( (v32 & 1) != 0 )
    goto LABEL_20;
  return a1;
}
