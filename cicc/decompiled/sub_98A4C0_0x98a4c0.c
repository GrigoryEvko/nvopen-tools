// Function: sub_98A4C0
// Address: 0x98a4c0
//
__int64 __fastcall sub_98A4C0(
        __int64 a1,
        unsigned int *a2,
        __int64 a3,
        __int64 a4,
        int a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9)
{
  __int64 v9; // r13
  signed __int64 v10; // r14
  __int64 v11; // r15
  unsigned int *v12; // r12
  unsigned int *v13; // rdx
  char v14; // al
  __int64 v15; // rax
  __int64 result; // rax
  unsigned int v17; // edx
  __int64 v18; // rcx
  unsigned __int64 v19; // rsi
  __int64 v20; // rax
  const void *v21; // r9
  unsigned __int64 v22; // rdx
  size_t v23; // r10
  int v24; // r15d
  int v25; // ecx
  __int64 v26; // r15
  __int64 v27; // rax
  unsigned __int64 v28; // rdx
  _BYTE *v29; // rsi
  _BYTE *v30; // rdi
  _DWORD *v31; // rax
  _DWORD *v32; // rcx
  __int64 v33; // rax
  int v34; // r15d
  __int64 v35; // rax
  __int64 v36; // r8
  __int64 v37; // r10
  _QWORD *v38; // r11
  _BYTE *v39; // rdi
  __int64 v40; // [rsp+0h] [rbp-A0h]
  int v41; // [rsp+8h] [rbp-98h]
  __int64 v42; // [rsp+10h] [rbp-90h]
  __int64 v43; // [rsp+18h] [rbp-88h]
  const void *v44; // [rsp+20h] [rbp-80h]
  __int64 v45; // [rsp+28h] [rbp-78h]
  _BYTE *v46; // [rsp+30h] [rbp-70h] BYREF
  __int64 v47; // [rsp+38h] [rbp-68h]
  _BYTE v48[96]; // [rsp+40h] [rbp-60h] BYREF

LABEL_1:
  v9 = a1;
  if ( !a3 )
    return a1;
  v10 = 4 * a3;
  v11 = a3;
  v12 = a2;
  v13 = &a2[a3];
  while ( 1 )
  {
    v14 = *(_BYTE *)v9;
    if ( *(_BYTE *)v9 <= 0x15u )
    {
      v15 = sub_AD69F0(v9, *v12);
      a2 = v12 + 1;
      a3 = v11 - 1;
      a1 = v15;
      if ( v15 )
        goto LABEL_1;
      return 0;
    }
    if ( v14 != 94 )
    {
      if ( v14 != 93 )
        return 0;
      v17 = *(_DWORD *)(v9 + 80);
      v18 = 0;
      v46 = v48;
      v19 = 5;
      v47 = 0x500000000LL;
      if ( v17 + (unsigned int)v11 > 5uLL )
      {
        sub_C8D5F0(&v46, v48, v17 + (unsigned int)v11, 4);
        v17 = *(_DWORD *)(v9 + 80);
        v18 = (unsigned int)v47;
        v19 = HIDWORD(v47);
      }
      v20 = v17;
      v21 = *(const void **)(v9 + 72);
      v22 = v18 + v17;
      v23 = 4 * v20;
      v24 = v20;
      if ( v22 > v19 )
      {
        v44 = *(const void **)(v9 + 72);
        v43 = 4 * v20;
        sub_C8D5F0(&v46, v48, v22, 4);
        v18 = (unsigned int)v47;
        v23 = v43;
        v21 = v44;
      }
      if ( v23 )
      {
        memcpy(&v46[4 * v18], v21, v23);
        LODWORD(v18) = v47;
      }
      LODWORD(v27) = v24 + v18;
      v25 = HIDWORD(v47);
      LODWORD(v47) = v27;
      v26 = v10 >> 2;
      v27 = (unsigned int)v27;
      v28 = (unsigned int)v27 + (v10 >> 2);
      if ( v28 > HIDWORD(v47) )
      {
        sub_C8D5F0(&v46, v48, v28, 4);
        v27 = (unsigned int)v47;
      }
      v29 = v46;
      if ( v10 )
      {
        memcpy(&v46[4 * v27], v12, v10);
        v29 = v46;
        LODWORD(v27) = v47;
      }
      LODWORD(v47) = v26 + v27;
      result = sub_98A4C0(*(_QWORD *)(v9 - 32), (_DWORD)v29, (int)v26 + (int)v27, v25, a5, (_DWORD)v21, a7, a8, a9);
      v30 = v46;
      if ( v46 == v48 )
        return result;
LABEL_21:
      v45 = result;
      _libc_free(v30, v29);
      return v45;
    }
    v31 = *(_DWORD **)(v9 + 72);
    v32 = &v31[*(unsigned int *)(v9 + 80)];
    if ( v32 == v31 )
    {
      a2 = v12;
LABEL_31:
      a1 = *(_QWORD *)(v9 - 32);
      a3 = v13 - a2;
      goto LABEL_1;
    }
    if ( v12 == v13 )
      break;
    a2 = v12;
    while ( *a2 == *v31 )
    {
      ++v31;
      ++a2;
      if ( v32 == v31 )
        goto LABEL_31;
      if ( a2 == v13 )
        goto LABEL_32;
    }
    v9 = *(_QWORD *)(v9 - 64);
  }
LABEL_32:
  if ( !(_BYTE)a9 )
    return 0;
  v33 = sub_B501B0(*(_QWORD *)(v9 + 8), v12, v10 >> 2);
  v34 = v33;
  v35 = sub_ACADE0(v33);
  v46 = v48;
  v36 = v10 >> 2;
  v37 = v35;
  v38 = &v46;
  v47 = 0xA00000000LL;
  if ( (unsigned __int64)v10 > 0x28 )
  {
    v40 = v35;
    sub_C8D5F0(&v46, v48, v10 >> 2, 4);
    v36 = v10 >> 2;
    v37 = v40;
    v39 = &v46[4 * (unsigned int)v47];
  }
  else
  {
    if ( !v10 )
      goto LABEL_35;
    v39 = v48;
  }
  v41 = v36;
  v42 = v37;
  memcpy(v39, v12, v10);
  LODWORD(v10) = v47;
  LODWORD(v38) = (unsigned int)&v46;
  LODWORD(v36) = v41;
  v37 = v42;
LABEL_35:
  v29 = (_BYTE *)v37;
  LODWORD(v47) = v10 + v36;
  result = sub_98A890(v9, v37, v34, (_DWORD)v38, (int)v10 + (int)v36, a7, a8, SBYTE1(a8));
  v30 = v46;
  if ( v46 != v48 )
    goto LABEL_21;
  return result;
}
