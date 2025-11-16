// Function: sub_D62CA0
// Address: 0xd62ca0
//
__int64 __fastcall sub_D62CA0(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rax
  unsigned int v9; // ecx
  unsigned int v10; // r13d
  __int64 v11; // rax
  __int64 v12; // rdx
  unsigned int v13; // r12d
  _QWORD *v14; // rdi
  char *v15; // r12
  __int64 v16; // rsi
  char *v17; // rbx
  __int64 v18; // rdi
  __int64 v19; // rdi
  _QWORD *v23; // [rsp+10h] [rbp-220h] BYREF
  unsigned int v24; // [rsp+18h] [rbp-218h]
  _QWORD *v25; // [rsp+20h] [rbp-210h] BYREF
  unsigned int v26; // [rsp+28h] [rbp-208h]
  __int64 v27; // [rsp+30h] [rbp-200h] BYREF
  unsigned int v28; // [rsp+38h] [rbp-1F8h]
  _QWORD *v29; // [rsp+40h] [rbp-1F0h] BYREF
  unsigned int v30; // [rsp+48h] [rbp-1E8h]
  _QWORD *v31; // [rsp+50h] [rbp-1E0h] BYREF
  unsigned int v32; // [rsp+58h] [rbp-1D8h]
  __int64 v33; // [rsp+60h] [rbp-1D0h] BYREF
  unsigned int v34; // [rsp+68h] [rbp-1C8h]
  _QWORD v35[5]; // [rsp+70h] [rbp-1C0h] BYREF
  __int64 v36; // [rsp+98h] [rbp-198h]
  unsigned int v37; // [rsp+A0h] [rbp-190h]
  char v38; // [rsp+B0h] [rbp-180h]
  char *v39; // [rsp+B8h] [rbp-178h] BYREF
  unsigned int v40; // [rsp+C0h] [rbp-170h]
  char v41; // [rsp+1F8h] [rbp-38h] BYREF

  v8 = sub_BD5C60(a1);
  sub_D5D740(v35, a3, a4, v8, a5, a6);
  sub_D62AB0((__int64)&v31, (__int64)v35, a1);
  if ( v32 <= 1 )
  {
    v10 = 0;
    if ( v34 <= 0x40 )
      goto LABEL_21;
    goto LABEL_44;
  }
  v9 = v34;
  v10 = 0;
  if ( v34 <= 1 )
    goto LABEL_21;
  v26 = v32;
  if ( v32 > 0x40 )
  {
    sub_C43780((__int64)&v25, (const void **)&v31);
    v9 = v34;
  }
  else
  {
    v25 = v31;
  }
  v28 = v9;
  if ( v9 > 0x40 )
  {
    sub_C43780((__int64)&v27, (const void **)&v33);
    v11 = 1LL << ((unsigned __int8)v28 - 1);
    if ( v28 > 0x40 )
    {
      v12 = *(_QWORD *)(v27 + 8LL * ((v28 - 1) >> 6));
      goto LABEL_8;
    }
  }
  else
  {
    v27 = v33;
    v11 = 1LL << ((unsigned __int8)v9 - 1);
  }
  v12 = v27;
LABEL_8:
  v13 = v26;
  if ( (v11 & v12) != 0 || (int)sub_C49970((__int64)&v25, (unsigned __int64 *)&v27) < 0 )
  {
    v24 = v13;
    if ( v13 > 0x40 )
      sub_C43690((__int64)&v23, 0, 0);
    else
      v23 = 0;
  }
  else
  {
    v30 = v13;
    if ( v13 > 0x40 )
      sub_C43780((__int64)&v29, (const void **)&v25);
    else
      v29 = v25;
    sub_C46B40((__int64)&v29, &v27);
    v24 = v30;
    v23 = v29;
  }
  if ( v28 > 0x40 && v27 )
    j_j___libc_free_0_0(v27);
  if ( v26 > 0x40 && v25 )
    j_j___libc_free_0_0(v25);
  v14 = v23;
  if ( v24 <= 0x40 )
  {
    *a2 = v23;
  }
  else
  {
    *a2 = *v23;
    j_j___libc_free_0_0(v14);
  }
  v10 = 1;
  if ( v34 > 0x40 )
  {
LABEL_44:
    if ( v33 )
      j_j___libc_free_0_0(v33);
  }
LABEL_21:
  if ( v32 > 0x40 && v31 )
    j_j___libc_free_0_0(v31);
  if ( (v38 & 1) != 0 )
  {
    v17 = &v41;
    v15 = (char *)&v39;
  }
  else
  {
    v15 = v39;
    v16 = 40LL * v40;
    if ( !v40 )
      goto LABEL_42;
    v17 = &v39[v16];
    if ( &v39[v16] == v39 )
      goto LABEL_42;
  }
  do
  {
    if ( *(_QWORD *)v15 != -8192 && *(_QWORD *)v15 != -4096 )
    {
      if ( *((_DWORD *)v15 + 8) > 0x40u )
      {
        v18 = *((_QWORD *)v15 + 3);
        if ( v18 )
          j_j___libc_free_0_0(v18);
      }
      if ( *((_DWORD *)v15 + 4) > 0x40u )
      {
        v19 = *((_QWORD *)v15 + 1);
        if ( v19 )
          j_j___libc_free_0_0(v19);
      }
    }
    v15 += 40;
  }
  while ( v17 != v15 );
  if ( (v38 & 1) == 0 )
  {
    v15 = v39;
    v16 = 40LL * v40;
LABEL_42:
    sub_C7D6A0((__int64)v15, v16, 8);
  }
  if ( v37 > 0x40 && v36 )
    j_j___libc_free_0_0(v36);
  return v10;
}
