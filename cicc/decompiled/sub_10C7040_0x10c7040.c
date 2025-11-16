// Function: sub_10C7040
// Address: 0x10c7040
//
__int64 __fastcall sub_10C7040(__int64 *a1, unsigned __int8 *a2)
{
  __int64 v3; // rdi
  int v4; // r15d
  bool v5; // al
  int v6; // r14d
  __int64 *v9; // rax
  __int64 v10; // r8
  unsigned int **v11; // r14
  const char *v12; // rax
  __int64 **v13; // rdx
  __int64 v14; // rdi
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // rax
  unsigned __int8 v19; // r10
  __int64 v20; // r14
  _BYTE *v21; // rdi
  bool v22; // al
  __int64 *v23; // r14
  const char *v24; // rax
  __int64 **v25; // rdx
  __int64 v26; // rax
  __int64 v27; // rax
  unsigned __int8 v28; // [rsp+Fh] [rbp-91h]
  __int64 v29; // [rsp+10h] [rbp-90h]
  __int64 *v30; // [rsp+10h] [rbp-90h]
  unsigned __int8 v31; // [rsp+18h] [rbp-88h]
  __int64 *v32; // [rsp+18h] [rbp-88h]
  unsigned __int8 v33; // [rsp+18h] [rbp-88h]
  __int64 *v34; // [rsp+28h] [rbp-78h] BYREF
  __int64 *v35; // [rsp+30h] [rbp-70h] BYREF
  int v36; // [rsp+38h] [rbp-68h]
  __int64 **v37; // [rsp+40h] [rbp-60h] BYREF
  __int64 **v38; // [rsp+48h] [rbp-58h]
  const char *v39; // [rsp+50h] [rbp-50h]
  __int64 **v40; // [rsp+58h] [rbp-48h]
  __int16 v41; // [rsp+60h] [rbp-40h]

  v37 = &v34;
  v38 = &v35;
  v39 = (const char *)&v34;
  v40 = &v35;
  if ( !sub_10C4D50(&v37, a2) || v34 == v35 )
    return 0;
  v3 = *((_QWORD *)a2 + 1);
  if ( (unsigned int)*(unsigned __int8 *)(v3 + 8) - 17 <= 1 )
    v3 = **(_QWORD **)(v3 + 16);
  v4 = 28;
  v5 = sub_BCAC40(v3, 1);
  v6 = *a2;
  if ( v5 )
  {
    v4 = 29;
    if ( (_BYTE)v6 != 57 )
    {
      v4 = 28;
      if ( (_BYTE)v6 == 86 && *(_QWORD *)(*((_QWORD *)a2 - 12) + 8LL) == *((_QWORD *)a2 + 1) )
      {
        v21 = (_BYTE *)*((_QWORD *)a2 - 4);
        if ( *v21 <= 0x15u )
        {
          v22 = sub_AC30F0((__int64)v21);
          v6 = *a2;
          if ( v22 )
            v4 = 29;
        }
      }
    }
  }
  if ( (unsigned __int8)sub_10C2350((__int64)a2, 0)
    && (unsigned __int8)sub_10C24F0(a1, v34, a2)
    && (v31 = sub_10C24F0(a1, v35, a2)) != 0 )
  {
    v34 = sub_10BFAB0((__int64)a1, (__int64)v34, a2);
    v9 = sub_10BFAB0((__int64)a1, (__int64)v35, a2);
    v10 = a1[4];
    v35 = v9;
    v29 = v10;
    sub_B445D0((__int64)&v37, (char *)a2);
    sub_10BF960(v29, (__int64)v37, (__int16)v38);
    if ( (unsigned int)(v6 - 42) <= 0x11 )
    {
      v23 = (__int64 *)a1[4];
      v24 = sub_BD5D20((__int64)a2);
      v38 = v25;
      v41 = 773;
      v37 = (__int64 **)v24;
      v39 = ".not";
      v26 = sub_10BBE20(v23, v4, (__int64)v34, (__int64)v35, v36, 0, (__int64)&v37, 0);
      v19 = v31;
      v20 = v26;
    }
    else
    {
      v11 = (unsigned int **)a1[4];
      v12 = sub_BD5D20((__int64)a2);
      v38 = v13;
      v37 = (__int64 **)v12;
      v41 = 773;
      v39 = ".not";
      v14 = v35[1];
      v28 = v31;
      v30 = v34;
      v32 = v35;
      if ( v4 == 29 )
      {
        v27 = sub_AD62B0(v14);
        v17 = (__int64)v32;
        v16 = v27;
      }
      else
      {
        v15 = sub_AD6530(v14, (__int64)v34);
        v16 = (__int64)v32;
        v17 = v15;
      }
      v18 = sub_B36550(v11, (__int64)v30, v16, v17, (__int64)&v37, 0);
      v19 = v28;
      v20 = v18;
    }
    v33 = v19;
    sub_F162A0((__int64)a1, (__int64)a2, v20);
    sub_F16650((__int64)a1, v20, 0);
    return v33;
  }
  else
  {
    return 0;
  }
}
