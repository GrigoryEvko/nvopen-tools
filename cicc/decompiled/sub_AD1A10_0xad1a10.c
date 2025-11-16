// Function: sub_AD1A10
// Address: 0xad1a10
//
__int64 __fastcall sub_AD1A10(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 v3; // r10
  __int64 *v5; // r8
  int v6; // edx
  unsigned __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 *v9; // rax
  __int64 *v10; // r13
  __int64 v11; // rdx
  __int64 *v12; // rbx
  char v13; // r14
  int v14; // r11d
  __int64 *v15; // r15
  __int64 v16; // r8
  __int64 v17; // r12
  __int64 v18; // rdx
  __int64 v19; // r12
  bool v21; // al
  _QWORD *v22; // rax
  __int64 v23; // [rsp+8h] [rbp-B8h]
  __int64 v24; // [rsp+10h] [rbp-B0h]
  __int64 *v25; // [rsp+18h] [rbp-A8h]
  __int64 v26; // [rsp+18h] [rbp-A8h]
  __int64 *v27; // [rsp+18h] [rbp-A8h]
  int v28; // [rsp+20h] [rbp-A0h]
  __int64 v29; // [rsp+20h] [rbp-A0h]
  __int64 v30; // [rsp+20h] [rbp-A0h]
  __int64 v31; // [rsp+20h] [rbp-A0h]
  int v32; // [rsp+28h] [rbp-98h]
  int v33; // [rsp+28h] [rbp-98h]
  __int64 *v34; // [rsp+28h] [rbp-98h]
  __int64 v35; // [rsp+30h] [rbp-90h]
  unsigned __int8 *v36; // [rsp+30h] [rbp-90h]
  unsigned int v37; // [rsp+38h] [rbp-88h]
  __int64 *v38; // [rsp+40h] [rbp-80h] BYREF
  __int64 v39; // [rsp+48h] [rbp-78h]
  _BYTE v40[112]; // [rsp+50h] [rbp-70h] BYREF

  v3 = a1;
  v5 = a2;
  v6 = *(_DWORD *)(a1 + 4);
  v38 = (__int64 *)v40;
  v39 = 0x800000000LL;
  v7 = v6 & 0x7FFFFFF;
  if ( v7 > 8 )
  {
    v34 = a2;
    a2 = (__int64 *)v40;
    v31 = a3;
    sub_C8D5F0(&v38, v40, v7, 8);
    v3 = a1;
    a3 = v31;
    v5 = v34;
    v7 = *(_DWORD *)(a1 + 4) & 0x7FFFFFF;
  }
  v8 = 4 * v7;
  if ( (*(_BYTE *)(v3 + 7) & 0x40) == 0 )
  {
    v10 = (__int64 *)v3;
    v9 = (__int64 *)(v3 - v8 * 8);
    if ( v3 - v8 * 8 != v3 )
      goto LABEL_5;
LABEL_18:
    v37 = 0;
    v14 = 0;
    goto LABEL_19;
  }
  v9 = *(__int64 **)(v3 - 8);
  v10 = &v9[v8];
  if ( v9 == &v9[v8] )
    goto LABEL_18;
LABEL_5:
  a2 = (__int64 *)v40;
  v11 = (unsigned int)v39;
  v12 = v9;
  v13 = 1;
  v37 = 0;
  v14 = 0;
  v15 = v5;
  do
  {
    v17 = *v12;
    if ( (__int64 *)*v12 == v15 )
    {
      v16 = v11 + 1;
      ++v14;
      v17 = a3;
      v37 = ((char *)v12 - (char *)v9) >> 5;
      if ( v11 + 1 <= (unsigned __int64)HIDWORD(v39) )
        goto LABEL_7;
    }
    else
    {
      v16 = v11 + 1;
      v13 &= a3 == v17;
      if ( v11 + 1 <= (unsigned __int64)HIDWORD(v39) )
        goto LABEL_7;
    }
    v23 = a3;
    v24 = v3;
    v25 = v9;
    v28 = v14;
    sub_C8D5F0(&v38, v40, v16, 8);
    v11 = (unsigned int)v39;
    a3 = v23;
    v3 = v24;
    v9 = v25;
    v14 = v28;
LABEL_7:
    v12 += 4;
    v38[v11] = v17;
    v11 = (unsigned int)(v39 + 1);
    LODWORD(v39) = v39 + 1;
  }
  while ( v12 != v10 );
  v5 = v15;
  if ( !v13 )
  {
LABEL_12:
    a2 = v38;
    v26 = a3;
    v29 = (__int64)v5;
    v32 = v14;
    v35 = v3;
    v19 = sub_ACB110(*(__int64 ***)(v3 + 8), (__int64)v38, v11);
    if ( !v19 )
    {
      v22 = (_QWORD *)sub_BD5C60(v35, a2, v18);
      a2 = v38;
      v19 = sub_AD1360(*v22 + 1744LL, v38, (unsigned int)v39, v35, v29, v26, v32, v37);
    }
    goto LABEL_14;
  }
LABEL_19:
  v27 = v5;
  v30 = v3;
  v33 = v14;
  v36 = (unsigned __int8 *)a3;
  v21 = sub_AC30F0(a3);
  a3 = (__int64)v36;
  v14 = v33;
  v3 = v30;
  v5 = v27;
  if ( v21 )
  {
    v19 = sub_AC9350(*(__int64 ***)(v30 + 8));
  }
  else
  {
    if ( (unsigned int)*v36 - 12 > 1 )
    {
      v11 = (unsigned int)v39;
      goto LABEL_12;
    }
    v19 = sub_ACA8A0(*(__int64 ***)(v30 + 8));
  }
LABEL_14:
  if ( v38 != (__int64 *)v40 )
    _libc_free(v38, a2);
  return v19;
}
