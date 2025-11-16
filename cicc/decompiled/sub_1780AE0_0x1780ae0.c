// Function: sub_1780AE0
// Address: 0x1780ae0
//
__int64 __fastcall sub_1780AE0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, double a5, double a6, double a7)
{
  __int64 v8; // r12
  unsigned __int8 v10; // al
  __int64 v11; // r15
  __int64 v12; // rax
  _BYTE *v13; // rsi
  __int64 *v14; // r10
  unsigned int *v15; // rax
  __int64 v16; // r11
  __int64 v17; // rax
  unsigned __int8 *v18; // r8
  unsigned __int8 *v19; // r9
  __int64 v20; // r12
  __int64 **v21; // rdx
  __int64 v22; // rax
  __int64 v23; // r12
  __int64 v25; // rax
  __int64 v26; // rdi
  __int64 v27; // r9
  __int64 *v28; // rbx
  __int64 v29; // rcx
  __int64 v30; // rax
  __int64 v31; // rdi
  __int64 v32; // rdx
  __int64 v33; // rsi
  __int64 v34; // rsi
  __int64 v35; // r12
  unsigned __int8 *v36; // rsi
  __int64 v37; // [rsp+8h] [rbp-98h]
  __int64 v38; // [rsp+10h] [rbp-90h]
  __int64 *v39; // [rsp+18h] [rbp-88h]
  __int64 v40; // [rsp+18h] [rbp-88h]
  __int64 v41; // [rsp+18h] [rbp-88h]
  __int64 v42; // [rsp+18h] [rbp-88h]
  __int64 v43; // [rsp+18h] [rbp-88h]
  unsigned __int8 *v44; // [rsp+18h] [rbp-88h]
  unsigned __int8 *v45; // [rsp+28h] [rbp-78h] BYREF
  __int64 v46[2]; // [rsp+30h] [rbp-70h] BYREF
  __int16 v47; // [rsp+40h] [rbp-60h]
  __int64 v48[2]; // [rsp+50h] [rbp-50h] BYREF
  __int16 v49; // [rsp+60h] [rbp-40h]

  v8 = a4;
  v10 = *(_BYTE *)(a2 + 16);
  v38 = a3;
  if ( v10 > 0x17u )
  {
    a3 = (unsigned int)v10 - 24;
  }
  else
  {
    if ( v10 != 5 )
      goto LABEL_3;
    a3 = *(unsigned __int16 *)(a2 + 18);
  }
  if ( (_DWORD)a3 == 37 )
  {
    if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    {
      a3 = *(_QWORD *)(a2 - 8);
    }
    else
    {
      a4 = 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
      a3 = a2 - a4;
    }
    v11 = *(_QWORD *)a3;
    if ( *(_QWORD *)a3 )
    {
      v10 = *(_BYTE *)(v11 + 16);
      if ( v10 != 47 )
        goto LABEL_4;
      goto LABEL_15;
    }
  }
LABEL_3:
  v11 = a2;
  if ( v10 != 47 )
  {
LABEL_4:
    if ( v10 != 5 )
      goto LABEL_43;
    if ( *(_WORD *)(v11 + 18) != 23 )
      goto LABEL_43;
    v12 = *(_DWORD *)(v11 + 20) & 0xFFFFFFF;
    a4 = 4 * v12;
    v13 = *(_BYTE **)(v11 - 24 * v12);
    if ( !v13 )
      goto LABEL_43;
    a3 = 1 - v12;
    v14 = *(__int64 **)(v11 + 24 * (1 - v12));
    if ( !v14 )
      goto LABEL_43;
    goto LABEL_17;
  }
LABEL_15:
  v13 = *(_BYTE **)(v11 - 48);
  if ( v13[16] > 0x10u || (v14 = *(__int64 **)(v11 - 24)) == 0 )
LABEL_43:
    BUG();
LABEL_17:
  v39 = v14;
  v15 = sub_177F600(*v14, v13, a3, a4);
  v16 = *(_QWORD *)(v8 + 8);
  v49 = 257;
  if ( *((_BYTE *)v39 + 16) > 0x10u || *((_BYTE *)v15 + 16) > 0x10u )
  {
    v19 = sub_170A2B0(v16, 11, v39, (__int64)v15, v48, 0, 0);
  }
  else
  {
    v37 = v16;
    v40 = sub_15A2B30(v39, (__int64)v15, 0, 0, a5, a6, a7);
    v17 = sub_14DBA30(v40, *(_QWORD *)(v37 + 96), 0);
    v18 = (unsigned __int8 *)v40;
    if ( v17 )
      v18 = (unsigned __int8 *)v17;
    v19 = v18;
  }
  if ( a2 != v11 )
  {
    v20 = *(_QWORD *)(v8 + 8);
    v21 = *(__int64 ***)a2;
    v47 = 257;
    if ( v21 != *(__int64 ***)v19 )
    {
      if ( v19[16] > 0x10u )
      {
        v49 = 257;
        v25 = sub_15FDBD0(37, (__int64)v19, (__int64)v21, (__int64)v48, 0);
        v26 = *(_QWORD *)(v20 + 8);
        v27 = v25;
        if ( v26 )
        {
          v28 = *(__int64 **)(v20 + 16);
          v42 = v25;
          sub_157E9D0(v26 + 40, v25);
          v27 = v42;
          v29 = *v28;
          v30 = *(_QWORD *)(v42 + 24);
          *(_QWORD *)(v42 + 32) = v28;
          v29 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v42 + 24) = v29 | v30 & 7;
          *(_QWORD *)(v29 + 8) = v42 + 24;
          *v28 = *v28 & 7 | (v42 + 24);
        }
        v31 = v27;
        v43 = v27;
        sub_164B780(v27, v46);
        v45 = (unsigned __int8 *)v43;
        if ( !*(_QWORD *)(v20 + 80) )
          sub_4263D6(v31, v46, v32);
        (*(void (__fastcall **)(__int64, unsigned __int8 **))(v20 + 88))(v20 + 64, &v45);
        v33 = *(_QWORD *)v20;
        v19 = (unsigned __int8 *)v43;
        if ( *(_QWORD *)v20 )
        {
          v45 = *(unsigned __int8 **)v20;
          sub_1623A60((__int64)&v45, v33, 2);
          v19 = (unsigned __int8 *)v43;
          v34 = *(_QWORD *)(v43 + 48);
          v35 = v43 + 48;
          if ( v34 )
          {
            sub_161E7C0(v43 + 48, v34);
            v19 = (unsigned __int8 *)v43;
          }
          v36 = v45;
          *((_QWORD *)v19 + 6) = v45;
          if ( v36 )
          {
            v44 = v19;
            sub_1623210((__int64)&v45, v36, v35);
            v19 = v44;
          }
        }
      }
      else
      {
        v41 = sub_15A46C0(37, (__int64 ***)v19, v21, 0);
        v22 = sub_14DBA30(v41, *(_QWORD *)(v20 + 96), 0);
        v19 = (unsigned __int8 *)v41;
        if ( v22 )
          v19 = (unsigned __int8 *)v22;
      }
    }
  }
  v49 = 257;
  v23 = sub_15FB440(24, a1, (__int64)v19, (__int64)v48, 0);
  if ( sub_15F23D0(v38) )
    sub_15F2350(v23, 1);
  return v23;
}
