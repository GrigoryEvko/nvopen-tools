// Function: sub_AD2CE0
// Address: 0xad2ce0
//
__int64 __fastcall sub_AD2CE0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r10
  unsigned __int64 v6; // rdx
  __int64 *v7; // rax
  __int64 v8; // rsi
  __int64 v9; // rdx
  __int64 *v10; // r13
  __int64 v11; // rdx
  __int64 *v12; // rbx
  char v13; // r14
  int v14; // r11d
  __int64 v15; // r15
  __int64 v16; // r8
  __int64 v17; // r12
  _QWORD *v18; // rax
  __int64 v19; // r12
  __int64 v21; // rdx
  bool v22; // al
  __int64 v23; // [rsp+8h] [rbp-B8h]
  __int64 v24; // [rsp+10h] [rbp-B0h]
  __int64 *v25; // [rsp+18h] [rbp-A8h]
  __int64 v26; // [rsp+18h] [rbp-A8h]
  __int64 v27; // [rsp+18h] [rbp-A8h]
  __int64 v28; // [rsp+18h] [rbp-A8h]
  int v29; // [rsp+20h] [rbp-A0h]
  __int64 v30; // [rsp+20h] [rbp-A0h]
  __int64 v31; // [rsp+20h] [rbp-A0h]
  __int64 v32; // [rsp+20h] [rbp-A0h]
  int v33; // [rsp+28h] [rbp-98h]
  int v34; // [rsp+28h] [rbp-98h]
  __int64 v35; // [rsp+30h] [rbp-90h]
  __int64 *v36; // [rsp+30h] [rbp-90h]
  unsigned __int8 *v37; // [rsp+30h] [rbp-90h]
  unsigned int v38; // [rsp+38h] [rbp-88h]
  __int64 *v39; // [rsp+40h] [rbp-80h] BYREF
  __int64 v40; // [rsp+48h] [rbp-78h]
  _BYTE v41[112]; // [rsp+50h] [rbp-70h] BYREF

  v4 = a1;
  v6 = *(_DWORD *)(a1 + 4) & 0x7FFFFFF;
  if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
    v7 = *(__int64 **)(a1 - 8);
  else
    v7 = (__int64 *)(a1 - 32 * v6);
  v8 = 0x800000000LL;
  v39 = (__int64 *)v41;
  v40 = 0x800000000LL;
  if ( v6 > 8 )
  {
    v8 = (__int64)v41;
    v27 = a3;
    v31 = a2;
    v36 = v7;
    sub_C8D5F0(&v39, v41, v6, 8);
    v4 = a1;
    v7 = v36;
    a3 = v27;
    a2 = v31;
    v21 = 4LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
    v10 = &v36[v21];
    if ( &v36[v21] != v36 )
      goto LABEL_5;
LABEL_18:
    v38 = 0;
    v14 = 0;
    goto LABEL_19;
  }
  v9 = 4 * v6;
  v10 = &v7[v9];
  if ( &v7[v9] == v7 )
    goto LABEL_18;
LABEL_5:
  v8 = (__int64)v41;
  v11 = (unsigned int)v40;
  v12 = v7;
  v13 = 1;
  v38 = 0;
  v14 = 0;
  v15 = a2;
  do
  {
    v17 = *v12;
    if ( *v12 == v15 )
    {
      v16 = v11 + 1;
      ++v14;
      v17 = a3;
      v38 = ((char *)v12 - (char *)v7) >> 5;
      if ( v11 + 1 <= (unsigned __int64)HIDWORD(v40) )
        goto LABEL_7;
    }
    else
    {
      v16 = v11 + 1;
      v13 &= a3 == v17;
      if ( v11 + 1 <= (unsigned __int64)HIDWORD(v40) )
        goto LABEL_7;
    }
    v23 = a3;
    v24 = v4;
    v25 = v7;
    v29 = v14;
    sub_C8D5F0(&v39, v41, v16, 8);
    v11 = (unsigned int)v40;
    a3 = v23;
    v4 = v24;
    v7 = v25;
    v14 = v29;
LABEL_7:
    v12 += 4;
    v39[v11] = v17;
    v11 = (unsigned int)(v40 + 1);
    LODWORD(v40) = v40 + 1;
  }
  while ( v10 != v12 );
  a2 = v15;
  if ( !v13 )
    goto LABEL_12;
LABEL_19:
  v28 = a2;
  v32 = v4;
  v34 = v14;
  v37 = (unsigned __int8 *)a3;
  v22 = sub_AC30F0(a3);
  a3 = (__int64)v37;
  v14 = v34;
  v4 = v32;
  a2 = v28;
  if ( v22 )
  {
    v19 = sub_AC9350(*(__int64 ***)(v32 + 8));
  }
  else if ( (unsigned int)*v37 - 12 > 1 )
  {
LABEL_12:
    v26 = a3;
    v30 = a2;
    v33 = v14;
    v35 = v4;
    v18 = (_QWORD *)sub_BD5C60(v4, v8, v11);
    v8 = (__int64)v39;
    v19 = sub_AD2630(*v18 + 1776LL, v39, (unsigned int)v40, v35, v30, v26, v33, v38);
  }
  else
  {
    v19 = sub_ACA8A0(*(__int64 ***)(v32 + 8));
  }
  if ( v39 != (__int64 *)v41 )
    _libc_free(v39, v8);
  return v19;
}
