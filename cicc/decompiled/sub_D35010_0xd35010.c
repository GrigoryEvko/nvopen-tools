// Function: sub_D35010
// Address: 0xd35010
//
__int64 __fastcall sub_D35010(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, char a7, char a8)
{
  __int64 v12; // rax
  unsigned int v13; // esi
  __int64 v14; // rax
  unsigned int v15; // esi
  _DWORD *v16; // rax
  __int64 v17; // r10
  unsigned int v18; // eax
  unsigned __int8 *v19; // rax
  __int64 v20; // r13
  __int64 v21; // rax
  unsigned int v22; // r13d
  int v23; // r13d
  __int64 v24; // rax
  __int64 v25; // rdx
  int v26; // ecx
  __int64 v27; // [rsp+0h] [rbp-A0h]
  unsigned __int8 *v28; // [rsp+8h] [rbp-98h]
  unsigned int v29; // [rsp+10h] [rbp-90h]
  __int64 v31; // [rsp+28h] [rbp-78h]
  char *v32; // [rsp+30h] [rbp-70h] BYREF
  unsigned int v33; // [rsp+38h] [rbp-68h]
  unsigned __int64 v34; // [rsp+40h] [rbp-60h] BYREF
  unsigned int v35; // [rsp+48h] [rbp-58h]
  unsigned __int64 v36; // [rsp+50h] [rbp-50h] BYREF
  __int64 v37; // [rsp+58h] [rbp-48h]
  char v38; // [rsp+60h] [rbp-40h]

  if ( a2 == a4 )
  {
    LODWORD(v31) = 0;
    BYTE4(v31) = 1;
    return v31;
  }
  if ( a1 != a3 && a8 )
    goto LABEL_4;
  v12 = *(_QWORD *)(a2 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v12 + 8) - 17 <= 1 )
    v12 = **(_QWORD **)(v12 + 16);
  v13 = *(_DWORD *)(v12 + 8);
  v14 = *(_QWORD *)(a4 + 8);
  v15 = v13 >> 8;
  if ( (unsigned int)*(unsigned __int8 *)(v14 + 8) - 17 <= 1 )
    v14 = **(_QWORD **)(v14 + 16);
  if ( *(_DWORD *)(v14 + 8) >> 8 != v15 )
  {
LABEL_4:
    BYTE4(v31) = 0;
    return v31;
  }
  v16 = sub_AE2980(a5, v15);
  v17 = a6;
  v18 = v16[3];
  v33 = v18;
  if ( v18 > 0x40 )
  {
    v29 = v18;
    sub_C43690((__int64)&v32, 0, 0);
    v35 = v29;
    sub_C43690((__int64)&v34, 0, 0);
    v17 = a6;
  }
  else
  {
    v35 = v18;
    v32 = 0;
    v34 = 0;
  }
  v27 = v17;
  v28 = sub_BD45C0((unsigned __int8 *)a2, a5, (__int64)&v32, 1, 0, 0, 0, 0);
  v19 = sub_BD45C0((unsigned __int8 *)a4, a5, (__int64)&v34, 1, 0, 0, 0, 0);
  if ( v19 == v28 )
  {
    v22 = sub_AE2980(a5, *(_DWORD *)(*((_QWORD *)v19 + 1) + 8LL) >> 8)[3];
    sub_C44B10((__int64)&v36, &v32, v22);
    if ( v33 > 0x40 && v32 )
      j_j___libc_free_0_0(v32);
    v32 = (char *)v36;
    v33 = v37;
    sub_C44B10((__int64)&v36, (char **)&v34, v22);
    if ( v35 > 0x40 && v34 )
      j_j___libc_free_0_0(v34);
    v34 = v36;
    v35 = v37;
    sub_C46B40((__int64)&v34, (__int64 *)&v32);
    if ( v35 > 0x40 )
    {
      v23 = *(_DWORD *)v34;
    }
    else
    {
      v23 = 0;
      if ( v35 )
        v23 = (__int64)(v34 << (64 - (unsigned __int8)v35)) >> (64 - (unsigned __int8)v35);
    }
  }
  else
  {
    v20 = sub_DD8400(v27, a2);
    v21 = sub_DD8400(v27, a4);
    sub_DC06D0(&v36, v27, v21, v20);
    if ( !v38 )
    {
LABEL_15:
      BYTE4(v31) = 0;
      goto LABEL_16;
    }
    if ( (unsigned int)v37 > 0x40 )
    {
      v23 = *(_DWORD *)v36;
      v38 = 0;
      j_j___libc_free_0_0(v36);
    }
    else
    {
      v23 = 0;
      if ( (_DWORD)v37 )
        v23 = (__int64)(v36 << (64 - (unsigned __int8)v37)) >> (64 - (unsigned __int8)v37);
    }
  }
  v24 = sub_9208B0(a5, a1);
  v37 = v25;
  v36 = (unsigned __int64)(v24 + 7) >> 3;
  v26 = sub_CA1930(&v36);
  if ( a7 && v23 / v26 * v26 != v23 )
    goto LABEL_15;
  LODWORD(v31) = v23 / v26;
  BYTE4(v31) = 1;
LABEL_16:
  if ( v35 > 0x40 && v34 )
    j_j___libc_free_0_0(v34);
  if ( v33 > 0x40 && v32 )
    j_j___libc_free_0_0(v32);
  return v31;
}
