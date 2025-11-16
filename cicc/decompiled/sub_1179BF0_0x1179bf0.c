// Function: sub_1179BF0
// Address: 0x1179bf0
//
__int64 __fastcall sub_1179BF0(__int64 a1, _BYTE *a2, _BYTE *a3, __int64 *a4)
{
  _BYTE *v6; // r12
  __int64 v7; // rbx
  __int64 v8; // rdi
  int v9; // edx
  __int16 v10; // r15
  unsigned int v11; // eax
  __int64 result; // rax
  char v13; // r15
  __int64 v14; // r15
  __int64 v15; // r10
  bool v16; // r12
  __int64 v17; // rdi
  __int64 v18; // rdx
  __int64 v19; // r10
  __int64 v20; // rbx
  __int64 v21; // r12
  __int64 v22; // rdx
  unsigned int v23; // esi
  __int64 v24; // r15
  __int64 v25; // rdx
  unsigned __int64 v26; // rdi
  char v27; // r15
  unsigned __int64 v28; // rdx
  char v29; // r15
  unsigned __int64 v30; // rdi
  char v31; // r15
  __int64 v32; // r15
  __int64 v33; // rdx
  _BYTE *v34; // rax
  _BYTE *v35; // rax
  __int64 v36; // rdi
  __int64 v37; // rdx
  __int64 *v38; // rbx
  __int64 v39; // r12
  __int64 v40; // rbx
  __int64 i; // r12
  __int64 v42; // rdx
  unsigned int v43; // esi
  unsigned int v44; // ecx
  _BYTE *v45; // rax
  unsigned int v46; // ecx
  char v47; // al
  unsigned int v48; // ecx
  _BYTE *v49; // rax
  unsigned int v50; // ecx
  char v51; // al
  int v52; // [rsp+8h] [rbp-A8h]
  int v53; // [rsp+8h] [rbp-A8h]
  unsigned int v54; // [rsp+Ch] [rbp-A4h]
  unsigned int v55; // [rsp+Ch] [rbp-A4h]
  __int64 v56; // [rsp+10h] [rbp-A0h]
  __int64 v57; // [rsp+10h] [rbp-A0h]
  __int64 v58; // [rsp+10h] [rbp-A0h]
  __int64 v60; // [rsp+18h] [rbp-98h]
  __int64 v61; // [rsp+18h] [rbp-98h]
  __int64 v62; // [rsp+18h] [rbp-98h]
  const char *v63; // [rsp+20h] [rbp-90h] BYREF
  __int64 v64; // [rsp+28h] [rbp-88h]
  __int16 v65; // [rsp+40h] [rbp-70h]
  unsigned __int64 v66; // [rsp+50h] [rbp-60h] BYREF
  unsigned int v67; // [rsp+58h] [rbp-58h]
  __int16 v68; // [rsp+70h] [rbp-40h]

  v6 = a2;
  v7 = *(_QWORD *)(a1 - 32);
  v8 = *(_QWORD *)(v7 + 8);
  v9 = *(unsigned __int8 *)(v8 + 8);
  if ( (unsigned int)(v9 - 17) <= 1 )
    LOBYTE(v9) = *(_BYTE *)(**(_QWORD **)(v8 + 16) + 8LL);
  if ( (_BYTE)v9 != 12 )
    return 0;
  v10 = *(_WORD *)(a1 + 2) & 0x3F;
  v56 = *(_QWORD *)(a1 - 64);
  v11 = sub_BCB060(v8);
  if ( v10 != 38 )
  {
    if ( v10 != 40 )
      return 0;
    v67 = v11;
    if ( v11 > 0x40 )
      sub_C43690((__int64)&v66, 0, 0);
    else
      v66 = 0;
    if ( *(_BYTE *)v7 == 17 )
    {
      v13 = sub_B532C0(v7 + 24, &v66, 39);
    }
    else
    {
      v24 = *(_QWORD *)(v7 + 8);
      v25 = (unsigned int)*(unsigned __int8 *)(v24 + 8) - 17;
      if ( (unsigned int)v25 > 1 || *(_BYTE *)v7 > 0x15u )
        goto LABEL_29;
      v35 = sub_AD7630(v7, 0, v25);
      if ( !v35 || *v35 != 17 )
      {
        if ( *(_BYTE *)(v24 + 8) == 17 )
        {
          v53 = *(_DWORD *)(v24 + 32);
          if ( v53 )
          {
            v13 = 0;
            v48 = 0;
            while ( 1 )
            {
              v55 = v48;
              v49 = (_BYTE *)sub_AD69F0((unsigned __int8 *)v7, v48);
              if ( !v49 )
                break;
              v50 = v55;
              if ( *v49 != 13 )
              {
                if ( *v49 != 17 )
                  break;
                v51 = sub_B532C0((__int64)(v49 + 24), &v66, 39);
                v50 = v55;
                v13 = v51;
                if ( !v51 )
                  break;
              }
              v48 = v50 + 1;
              if ( v53 == v48 )
                goto LABEL_12;
            }
          }
        }
        goto LABEL_29;
      }
      v13 = sub_B532C0((__int64)(v35 + 24), &v66, 39);
    }
LABEL_12:
    if ( v13 )
    {
      if ( v67 <= 0x40 || (v26 = v66, v27 = 0, !v66) )
      {
LABEL_14:
        v6 = a3;
        a3 = a2;
        goto LABEL_15;
      }
LABEL_32:
      j_j___libc_free_0_0(v26);
      if ( v27 )
        return 0;
      goto LABEL_14;
    }
LABEL_29:
    if ( v67 <= 0x40 )
      return 0;
    v26 = v66;
    if ( !v66 )
      return 0;
    v27 = 1;
    goto LABEL_32;
  }
  v67 = v11;
  if ( v11 > 0x40 )
  {
    sub_C43690((__int64)&v66, -1, 1);
  }
  else
  {
    v28 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v11;
    if ( !v11 )
      v28 = 0;
    v66 = v28;
  }
  if ( *(_BYTE *)v7 == 17 )
  {
    v29 = sub_B532C0(v7 + 24, &v66, 39);
  }
  else
  {
    v32 = *(_QWORD *)(v7 + 8);
    v33 = (unsigned int)*(unsigned __int8 *)(v32 + 8) - 17;
    if ( (unsigned int)v33 > 1 || *(_BYTE *)v7 > 0x15u )
      goto LABEL_45;
    v34 = sub_AD7630(v7, 0, v33);
    if ( !v34 || *v34 != 17 )
    {
      if ( *(_BYTE *)(v32 + 8) == 17 )
      {
        v52 = *(_DWORD *)(v32 + 32);
        if ( v52 )
        {
          v29 = 0;
          v44 = 0;
          while ( 1 )
          {
            v54 = v44;
            v45 = (_BYTE *)sub_AD69F0((unsigned __int8 *)v7, v44);
            if ( !v45 )
              break;
            v46 = v54;
            if ( *v45 != 13 )
            {
              if ( *v45 != 17 )
                break;
              v47 = sub_B532C0((__int64)(v45 + 24), &v66, 39);
              v46 = v54;
              v29 = v47;
              if ( !v47 )
                break;
            }
            v44 = v46 + 1;
            if ( v52 == v44 )
              goto LABEL_40;
          }
        }
      }
      goto LABEL_45;
    }
    v29 = sub_B532C0((__int64)(v34 + 24), &v66, 39);
  }
LABEL_40:
  if ( v29 )
  {
    if ( v67 <= 0x40 )
      goto LABEL_15;
    v30 = v66;
    v31 = 0;
    if ( !v66 )
      goto LABEL_15;
    goto LABEL_48;
  }
LABEL_45:
  if ( v67 <= 0x40 )
    return 0;
  v30 = v66;
  if ( !v66 )
    return 0;
  v31 = 1;
LABEL_48:
  j_j___libc_free_0_0(v30);
  if ( v31 )
    return 0;
LABEL_15:
  if ( *v6 != 55 )
    return 0;
  v14 = *((_QWORD *)v6 - 8);
  if ( !v14 )
    return 0;
  v15 = *((_QWORD *)v6 - 4);
  if ( !v15 || *a3 != 56 || v14 != *((_QWORD *)a3 - 8) || *((_QWORD *)a3 - 4) != v15 || v56 != v14 )
    return 0;
  v57 = *((_QWORD *)v6 - 4);
  if ( sub_B44E60((__int64)a3) )
  {
    v16 = sub_B44E60((__int64)v6);
    v63 = sub_BD5D20(a1);
    v65 = 261;
    v17 = a4[10];
    v64 = v18;
    result = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, bool))(*(_QWORD *)v17 + 24LL))(
               v17,
               27,
               v14,
               v57,
               v16);
    if ( result )
      return result;
    v19 = v57;
    if ( v16 )
    {
      v68 = 257;
      v58 = sub_B504D0(27, v14, v57, (__int64)&v66, 0, 0);
      sub_B448B0(v58, 1);
      (*(void (__fastcall **)(__int64, __int64, const char **, __int64, __int64))(*(_QWORD *)a4[11] + 16LL))(
        a4[11],
        v58,
        &v63,
        a4[7],
        a4[8]);
      result = v58;
      v20 = *a4;
      v21 = *a4 + 16LL * *((unsigned int *)a4 + 2);
      if ( *a4 != v21 )
      {
        do
        {
          v22 = *(_QWORD *)(v20 + 8);
          v23 = *(_DWORD *)v20;
          v20 += 16;
          v60 = result;
          sub_B99FD0(result, v23, v22);
          result = v60;
        }
        while ( v21 != v20 );
      }
      return result;
    }
  }
  else
  {
    v63 = sub_BD5D20(a1);
    v65 = 261;
    v36 = a4[10];
    v64 = v37;
    result = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, _QWORD))(*(_QWORD *)v36 + 24LL))(
               v36,
               27,
               v14,
               v57,
               0);
    v19 = v57;
    if ( result )
      return result;
  }
  v68 = 257;
  v38 = a4;
  v61 = sub_B504D0(27, v14, v19, (__int64)&v66, 0, 0);
  (*(void (__fastcall **)(__int64, __int64, const char **, __int64, __int64))(*(_QWORD *)v38[11] + 16LL))(
    v38[11],
    v61,
    &v63,
    v38[7],
    v38[8]);
  result = v61;
  v39 = 16LL * *((unsigned int *)v38 + 2);
  v40 = *v38;
  for ( i = v40 + v39; i != v40; result = v62 )
  {
    v42 = *(_QWORD *)(v40 + 8);
    v43 = *(_DWORD *)v40;
    v40 += 16;
    v62 = result;
    sub_B99FD0(result, v43, v42);
  }
  return result;
}
