// Function: sub_29D04A0
// Address: 0x29d04a0
//
__int64 __fastcall sub_29D04A0(__int64 *a1, unsigned __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v7; // r15
  __int64 v8; // rdx
  __int64 v9; // rax
  unsigned __int64 v10; // rbx
  unsigned __int64 v11; // rcx
  unsigned __int64 v12; // rax
  unsigned int v13; // r8d
  __int64 v15; // rdx
  char v16; // dl
  __int64 v17; // rax
  __int64 *v18; // rdx
  __int64 v19; // rsi
  int v20; // eax
  unsigned __int8 v21; // r8
  __int64 *v22; // rdx
  __int64 v23; // r12
  char v24; // al
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rax
  unsigned __int64 v29; // [rsp+8h] [rbp-88h]
  unsigned __int64 v30; // [rsp+10h] [rbp-80h]
  char v31; // [rsp+1Bh] [rbp-75h]
  unsigned int v32; // [rsp+1Ch] [rbp-74h]
  int v33; // [rsp+1Ch] [rbp-74h]
  unsigned __int8 v34; // [rsp+1Ch] [rbp-74h]
  __int64 v35; // [rsp+28h] [rbp-68h] BYREF
  __int64 v36; // [rsp+30h] [rbp-60h]
  __int64 v37; // [rsp+38h] [rbp-58h]
  unsigned __int64 v38; // [rsp+40h] [rbp-50h] BYREF
  __int64 v39; // [rsp+48h] [rbp-48h]
  char v40; // [rsp+50h] [rbp-40h]

  v7 = *(_QWORD *)(a2 + 8);
  v38 = sub_9208B0(a4, v7);
  v39 = v8;
  v30 = (v38 + 7) >> 3;
  v31 = v8;
LABEL_2:
  v32 = *(_DWORD *)(a3 + 8);
  if ( v32 <= 0x40 )
  {
LABEL_3:
    if ( !*(_QWORD *)a3 )
      goto LABEL_27;
    goto LABEL_4;
  }
  while ( 1 )
  {
    if ( v32 - (unsigned int)sub_C444A0(a3) > 0x40 || **(_QWORD **)a3 )
      goto LABEL_4;
LABEL_27:
    v18 = (__int64 *)(*a1 & 0xFFFFFFFFFFFFFFF8LL);
    if ( *a1 && (*a1 & 4) == 0 && v18 )
      v19 = v18[1];
    else
      v19 = *v18;
    v21 = sub_B50C50(v7, v19, a4);
    if ( v21 )
      break;
LABEL_4:
    v9 = *a1;
    if ( (*a1 & 4) == 0 )
    {
      if ( !(unsigned __int8)sub_29D02E0(a1) )
        return 0;
      v9 = *a1;
    }
    v10 = v9 & 0xFFFFFFFFFFFFFFF8LL;
    v35 = *(_QWORD *)(v9 & 0xFFFFFFFFFFFFFFF8LL);
    sub_AE5800((__int64)&v38, a4, &v35, (unsigned __int64 **)a3);
    if ( !v40 )
      return 0;
    v11 = *(unsigned int *)(v10 + 16);
    v33 = v39;
    if ( (unsigned int)v39 > 0x40 )
    {
      v29 = *(unsigned int *)(v10 + 16);
      v20 = sub_C444A0((__int64)&v38);
      v11 = v29;
      if ( (unsigned int)(v33 - v20) > 0x40 )
        goto LABEL_11;
      v12 = *(_QWORD *)v38;
    }
    else
    {
      v12 = v38;
    }
    if ( v11 <= v12 )
      goto LABEL_11;
    v36 = sub_9208B0(a4, v35);
    v37 = v15;
    if ( v31 && !(_BYTE)v15 )
    {
      v16 = v40;
LABEL_19:
      if ( v16 )
      {
LABEL_11:
        v40 = 0;
        if ( (unsigned int)v39 > 0x40 && v38 )
          j_j___libc_free_0_0(v38);
      }
      return 0;
    }
    v16 = v40;
    if ( (unsigned __int64)(v36 + 7) >> 3 < v30 )
      goto LABEL_19;
    v17 = *(_QWORD *)(v10 + 8);
    if ( (unsigned int)v39 <= 0x40 )
    {
      a1 = (__int64 *)(v17 + 8 * v38);
      goto LABEL_2;
    }
    a1 = (__int64 *)(v17 + 8LL * *(_QWORD *)v38);
    if ( !v40 )
      goto LABEL_2;
    v40 = 0;
    j_j___libc_free_0_0(v38);
    v32 = *(_DWORD *)(a3 + 8);
    if ( v32 <= 0x40 )
      goto LABEL_3;
  }
  v22 = (__int64 *)(*a1 & 0xFFFFFFFFFFFFFFF8LL);
  if ( *a1 && (*a1 & 4) == 0 && v22 )
    v23 = v22[1];
  else
    v23 = *v22;
  v34 = v21;
  sub_29CF750(a1);
  v24 = *(_BYTE *)(v7 + 8);
  v13 = v34;
  if ( v24 == 12 )
  {
    if ( *(_BYTE *)(v23 + 8) != 14 )
      goto LABEL_42;
    v26 = sub_AD4C70(a2, (__int64 **)v23, 0);
    v13 = v34;
    *a1 = v26 & 0xFFFFFFFFFFFFFFFBLL;
  }
  else if ( v24 == 14 && *(_BYTE *)(v23 + 8) == 12 )
  {
    v27 = sub_AD4C50(a2, (__int64 **)v23, 0);
    v13 = v34;
    *a1 = v27 & 0xFFFFFFFFFFFFFFFBLL;
  }
  else
  {
LABEL_42:
    if ( v7 == v23 )
    {
      *a1 = a2 & 0xFFFFFFFFFFFFFFFBLL;
    }
    else
    {
      v25 = sub_AD4C90(a2, (__int64 **)v23, 0);
      v13 = v34;
      *a1 = v25 & 0xFFFFFFFFFFFFFFFBLL;
    }
  }
  return v13;
}
