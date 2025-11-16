// Function: sub_317FCF0
// Address: 0x317fcf0
//
void __fastcall sub_317FCF0(_QWORD *a1)
{
  __int64 **v2; // rax
  __int64 v3; // r15
  __int64 v4; // rax
  __int64 v5; // rax
  int *v6; // rbx
  size_t v7; // rdx
  size_t v8; // r15
  unsigned __int64 v9; // rdi
  unsigned __int64 **v10; // r8
  unsigned __int64 *v11; // rax
  unsigned __int64 *v12; // rsi
  unsigned __int64 *v13; // rbx
  _BYTE *v14; // rsi
  __int64 v15; // r15
  __int64 v16; // rax
  __int64 v17; // r15
  __int64 i; // rbx
  __int64 v19; // rdx
  __int64 v20; // [rsp+10h] [rbp-1B0h] BYREF
  size_t v21; // [rsp+18h] [rbp-1A8h] BYREF
  unsigned __int64 *v22[2]; // [rsp+20h] [rbp-1A0h] BYREF
  unsigned __int64 v23[4]; // [rsp+30h] [rbp-190h] BYREF
  __int64 v24[2]; // [rsp+50h] [rbp-170h] BYREF
  __int64 *v25; // [rsp+60h] [rbp-160h]
  __int64 *v26; // [rsp+68h] [rbp-158h]
  __int64 v27; // [rsp+70h] [rbp-150h]
  __int64 **v28; // [rsp+78h] [rbp-148h]
  __int64 **v29; // [rsp+80h] [rbp-140h]
  __int64 v30; // [rsp+88h] [rbp-138h]
  __int64 v31; // [rsp+90h] [rbp-130h]
  __int64 v32; // [rsp+98h] [rbp-128h]
  __int64 v33[2]; // [rsp+A0h] [rbp-120h] BYREF
  _QWORD *v34; // [rsp+B0h] [rbp-110h]
  __int64 v35; // [rsp+B8h] [rbp-108h]
  __int64 v36; // [rsp+C0h] [rbp-100h]
  __int64 v37; // [rsp+C8h] [rbp-F8h]
  _QWORD *v38; // [rsp+D0h] [rbp-F0h]
  __int64 v39; // [rsp+D8h] [rbp-E8h]
  __int64 v40; // [rsp+E0h] [rbp-E0h]
  __int64 v41; // [rsp+E8h] [rbp-D8h]
  __int64 *v42[26]; // [rsp+F0h] [rbp-D0h] BYREF

  v42[0] = a1 + 15;
  v24[0] = 0;
  v24[1] = 0;
  v25 = 0;
  v26 = 0;
  v27 = 0;
  v28 = 0;
  v29 = 0;
  v30 = 0;
  v31 = 0;
  v32 = 0;
  sub_26C4970(v24, 0);
  v2 = v29;
  if ( v29 == (__int64 **)(v31 - 8) )
  {
    sub_26C4A60((unsigned __int64 *)v24, v42);
  }
  else
  {
    if ( v29 )
    {
      *v29 = v42[0];
      v2 = v29;
    }
    v29 = v2 + 1;
  }
  v33[0] = 0;
  v33[1] = 0;
  v34 = 0;
  v35 = 0;
  v36 = 0;
  v37 = 0;
  v38 = 0;
  v39 = 0;
  v40 = 0;
  v41 = 0;
  sub_26C4970(v33, 0);
  while ( v29 == (__int64 **)v25 )
  {
    if ( v38 == v34 )
      goto LABEL_33;
    v3 = *v25;
LABEL_9:
    v4 = sub_317E470(v3);
    v20 = v4;
    if ( !v4 )
      goto LABEL_24;
    *(_DWORD *)(v4 + 48) |= 1u;
    v42[0] = (__int64 *)v4;
    *(_QWORD *)sub_317EE30(a1 + 7, (unsigned __int64 *)v42) = v3;
    v5 = sub_317E460(v3);
    memset(v23, 0, 24);
    v6 = (int *)v5;
    v8 = v7;
    if ( v5 )
    {
      sub_C7D030(v42);
      sub_C7D280((int *)v42, v6, v8);
      sub_C7D290(v42, v22);
      v8 = (size_t)v22[0];
    }
    v9 = a1[1];
    v21 = v8;
    v10 = *(unsigned __int64 ***)(*a1 + 8 * (v8 % v9));
    if ( !v10 )
      goto LABEL_31;
    v11 = *v10;
    if ( v8 != (*v10)[1] )
    {
      while ( 1 )
      {
        v12 = (unsigned __int64 *)*v11;
        if ( !*v11 )
          break;
        v10 = (unsigned __int64 **)v11;
        if ( v8 % v9 != v12[1] % v9 )
          break;
        v11 = (unsigned __int64 *)*v11;
        if ( v8 == v12[1] )
          goto LABEL_17;
      }
LABEL_31:
      v42[0] = (__int64 *)v23;
      v22[0] = &v21;
      v13 = sub_317D460(a1, v22, v42);
      goto LABEL_18;
    }
LABEL_17:
    v13 = *v10;
    if ( !*v10 )
      goto LABEL_31;
LABEL_18:
    if ( v23[0] )
      j_j___libc_free_0(v23[0]);
    v14 = (_BYTE *)v13[3];
    if ( v14 == (_BYTE *)v13[4] )
    {
      sub_317F090((__int64)(v13 + 2), v14, &v20);
    }
    else
    {
      if ( v14 )
      {
        *(_QWORD *)v14 = v20;
        v14 = (_BYTE *)v13[3];
      }
      v13[3] = (unsigned __int64)(v14 + 8);
    }
LABEL_24:
    v15 = *v25;
    if ( v25 == (__int64 *)(v27 - 8) )
    {
      j_j___libc_free_0((unsigned __int64)v26);
      v19 = (__int64)(*++v28 + 64);
      v26 = *v28;
      v27 = v19;
      v25 = v26;
    }
    else
    {
      ++v25;
    }
    v16 = sub_317E450(v15);
    v17 = *(_QWORD *)(v16 + 24);
    for ( i = v16 + 8; i != v17; v17 = sub_220EEE0(v17) )
    {
      v42[0] = (__int64 *)(v17 + 40);
      sub_317FBD0((unsigned __int64 *)v24, v42);
    }
  }
  v3 = *v25;
  if ( v38 == v34 || *v34 != v3 )
    goto LABEL_9;
LABEL_33:
  sub_26C2C00((unsigned __int64 *)v33);
  sub_26C2C00((unsigned __int64 *)v24);
}
