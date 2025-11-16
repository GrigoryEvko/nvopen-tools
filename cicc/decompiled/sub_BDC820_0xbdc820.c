// Function: sub_BDC820
// Address: 0xbdc820
//
unsigned __int64 __fastcall sub_BDC820(__int64 a1, __int64 a2)
{
  __int64 v3; // r13
  _BYTE *v4; // r12
  unsigned __int64 result; // rax
  __int64 v6; // rcx
  __int64 *v7; // rdx
  char v8; // dl
  _QWORD *v9; // rdi
  __int64 v10; // r15
  unsigned __int8 v11; // al
  __int64 v12; // rax
  __int64 *v13; // rbx
  __int64 *v14; // r11
  __int64 *v15; // r15
  _BYTE *v16; // r10
  _QWORD *v17; // rax
  __int64 v18; // rcx
  _QWORD *v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rbx
  __int64 v22; // r13
  __int64 v23; // rcx
  __int64 v24; // rdi
  void *v25; // rdx
  __int64 v26; // rax
  _WORD *v27; // rdx
  __int64 v28; // rdi
  void *v29; // rdx
  __int64 v30; // rax
  _WORD *v31; // rdx
  __int64 v32; // rdi
  const char *v33; // rax
  __int64 v34; // rbx
  _BYTE *v35; // rax
  char v36; // dl
  __int64 v37; // rax
  unsigned __int64 v38; // rdx
  __int64 v39; // rax
  __int64 v40; // [rsp+10h] [rbp-100h]
  __int64 v41; // [rsp+10h] [rbp-100h]
  _BYTE *v42; // [rsp+10h] [rbp-100h]
  __int64 v43; // [rsp+10h] [rbp-100h]
  _BYTE *v44; // [rsp+10h] [rbp-100h]
  _QWORD v45[4]; // [rsp+20h] [rbp-F0h] BYREF
  char v46; // [rsp+40h] [rbp-D0h]
  char v47; // [rsp+41h] [rbp-CFh]
  _QWORD *v48; // [rsp+50h] [rbp-C0h] BYREF
  __int64 v49; // [rsp+58h] [rbp-B8h]
  _QWORD v50[22]; // [rsp+60h] [rbp-B0h] BYREF

  v3 = a1 + 944;
  v4 = (_BYTE *)a2;
  if ( !*(_BYTE *)(a1 + 972) )
    goto LABEL_8;
  result = *(_QWORD *)(a1 + 952);
  v6 = *(unsigned int *)(a1 + 964);
  v7 = (__int64 *)(result + 8 * v6);
  if ( (__int64 *)result != v7 )
  {
    while ( *(_QWORD *)result != a2 )
    {
      result += 8LL;
      if ( v7 == (__int64 *)result )
        goto LABEL_7;
    }
    return result;
  }
LABEL_7:
  if ( (unsigned int)v6 < *(_DWORD *)(a1 + 960) )
  {
    *(_DWORD *)(a1 + 964) = v6 + 1;
    *v7 = a2;
    ++*(_QWORD *)(a1 + 944);
  }
  else
  {
LABEL_8:
    result = sub_C8CC70(a1 + 944, a2);
    if ( !v8 )
      return result;
  }
  v9 = v50;
  v50[0] = a2;
  v48 = v50;
  v49 = 0x1000000001LL;
  LODWORD(result) = 1;
  while ( 1 )
  {
    v10 = v9[(unsigned int)result - 1];
    LODWORD(v49) = result - 1;
    v11 = *(_BYTE *)v10;
    if ( *(_BYTE *)v10 == 5 )
    {
      if ( *(_WORD *)(v10 + 2) != 49 )
        goto LABEL_12;
      a2 = *(_QWORD *)(*(_QWORD *)(v10 - 32LL * (*(_DWORD *)(v10 + 4) & 0x7FFFFFF)) + 8LL);
      if ( !(unsigned __int8)sub_B50F30(49, a2, *(_QWORD *)(v10 + 8)) )
      {
        a2 = (__int64)v45;
        v47 = 1;
        v45[0] = "Invalid bitcast";
        v46 = 3;
        sub_BDBF70((__int64 *)a1, (__int64)v45);
        if ( *(_QWORD *)a1 )
        {
          a2 = v10;
          sub_BDBD80(a1, (_BYTE *)v10);
        }
      }
      v11 = *(_BYTE *)v10;
    }
    if ( v11 == 8 )
    {
      v20 = *(_QWORD *)(*(_QWORD *)(v10 - 128) + 8LL);
      if ( *(_BYTE *)(v20 + 8) == 14 )
      {
        if ( v20 == *(_QWORD *)(v10 + 8) )
        {
          if ( *(_DWORD *)(*(_QWORD *)(v10 - 96) + 32LL) == 32 )
          {
            if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v10 - 32) + 8LL) + 8LL) == 14 )
            {
              if ( *(_DWORD *)(*(_QWORD *)(v10 - 64) + 32LL) == 64 )
                goto LABEL_12;
              a2 = (__int64)v45;
              v47 = 1;
              v45[0] = "signed ptrauth constant discriminator must be i64 constant integer";
              v46 = 3;
              sub_BDBF70((__int64 *)a1, (__int64)v45);
              v11 = *(_BYTE *)v10;
              goto LABEL_34;
            }
            v47 = 1;
            v33 = "signed ptrauth constant address discriminator must be a pointer";
          }
          else
          {
            v47 = 1;
            v33 = "signed ptrauth constant key must be i32 constant integer";
          }
        }
        else
        {
          v47 = 1;
          v33 = "signed ptrauth constant must have same type as its base pointer";
        }
      }
      else
      {
        v47 = 1;
        v33 = "signed ptrauth constant base pointer must have pointer type";
      }
      v34 = *(_QWORD *)a1;
      v45[0] = v33;
      v46 = 3;
      if ( v34 )
      {
        a2 = v34;
        sub_CA0E80(v45, v34);
        v35 = *(_BYTE **)(v34 + 32);
        if ( (unsigned __int64)v35 >= *(_QWORD *)(v34 + 24) )
        {
          a2 = 10;
          sub_CB5D20(v34, 10);
        }
        else
        {
          *(_QWORD *)(v34 + 32) = v35 + 1;
          *v35 = 10;
        }
      }
      *(_BYTE *)(a1 + 152) = 1;
      v11 = *(_BYTE *)v10;
    }
LABEL_34:
    if ( v11 > 3u )
    {
LABEL_12:
      v12 = 4LL * (*(_DWORD *)(v10 + 4) & 0x7FFFFFF);
      if ( (*(_BYTE *)(v10 + 7) & 0x40) != 0 )
      {
        v14 = *(__int64 **)(v10 - 8);
        v13 = &v14[v12];
      }
      else
      {
        v13 = (__int64 *)v10;
        v14 = (__int64 *)(v10 - v12 * 8);
      }
      v15 = v14;
      if ( v13 != v14 )
      {
        while ( 1 )
        {
          v16 = (_BYTE *)*v15;
          if ( *(_BYTE *)*v15 > 0x15u )
            goto LABEL_21;
          if ( *(_BYTE *)(a1 + 972) )
          {
            v17 = *(_QWORD **)(a1 + 952);
            v18 = *(unsigned int *)(a1 + 964);
            v19 = &v17[v18];
            if ( v17 != v19 )
            {
              while ( v16 != (_BYTE *)*v17 )
              {
                if ( v19 == ++v17 )
                  goto LABEL_66;
              }
              goto LABEL_21;
            }
LABEL_66:
            if ( (unsigned int)v18 < *(_DWORD *)(a1 + 960) )
            {
              *(_DWORD *)(a1 + 964) = v18 + 1;
              *v19 = v16;
              ++*(_QWORD *)(a1 + 944);
              goto LABEL_62;
            }
          }
          a2 = *v15;
          v42 = (_BYTE *)*v15;
          sub_C8CC70(v3, *v15);
          v16 = v42;
          if ( v36 )
          {
LABEL_62:
            v37 = (unsigned int)v49;
            v38 = (unsigned int)v49 + 1LL;
            if ( v38 > HIDWORD(v49) )
            {
              a2 = (__int64)v50;
              v44 = v16;
              sub_C8D5F0(&v48, v50, v38, 8);
              v37 = (unsigned int)v49;
              v16 = v44;
            }
            v15 += 4;
            v48[v37] = v16;
            LODWORD(v49) = v49 + 1;
            if ( v13 == v15 )
              goto LABEL_22;
          }
          else
          {
LABEL_21:
            v15 += 4;
            if ( v13 == v15 )
              goto LABEL_22;
          }
        }
      }
      goto LABEL_22;
    }
    v21 = *(_QWORD *)(v10 + 40);
    if ( *(_QWORD *)(a1 + 8) != v21 )
      break;
LABEL_22:
    result = (unsigned int)v49;
    v9 = v48;
    if ( !(_DWORD)v49 )
    {
      if ( v48 == v50 )
        return result;
      return _libc_free(v9, a2);
    }
  }
  v22 = *(_QWORD *)a1;
  result = (unsigned __int64)"Referencing global in another module!";
  v40 = *(_QWORD *)(a1 + 8);
  v47 = 1;
  v45[0] = "Referencing global in another module!";
  v46 = 3;
  if ( v22 )
  {
    a2 = v22;
    sub_CA0E80(v45, v22);
    result = *(_QWORD *)(v22 + 32);
    v23 = v40;
    if ( result >= *(_QWORD *)(v22 + 24) )
    {
      a2 = 10;
      result = sub_CB5D20(v22, 10);
      v24 = *(_QWORD *)a1;
      v23 = v40;
    }
    else
    {
      *(_QWORD *)(v22 + 32) = result + 1;
      *(_BYTE *)result = 10;
      v24 = *(_QWORD *)a1;
    }
    *(_BYTE *)(a1 + 152) = 1;
    if ( v24 )
    {
      if ( v4 )
      {
        v41 = v23;
        sub_BDBD80(a1, v4);
        v24 = *(_QWORD *)a1;
        v23 = v41;
      }
      v25 = *(void **)(v24 + 32);
      if ( *(_QWORD *)(v24 + 24) - (_QWORD)v25 <= 0xDu )
      {
        v43 = v23;
        v39 = sub_CB6200(v24, "; ModuleID = '", 14);
        v23 = v43;
        v24 = v39;
      }
      else
      {
        qmemcpy(v25, "; ModuleID = '", 14);
        *(_QWORD *)(v24 + 32) += 14LL;
      }
      v26 = sub_CB6200(v24, *(_QWORD *)(v23 + 168), *(_QWORD *)(v23 + 176));
      v27 = *(_WORD **)(v26 + 32);
      if ( *(_QWORD *)(v26 + 24) - (_QWORD)v27 <= 1u )
      {
        sub_CB6200(v26, "'\n", 2);
      }
      else
      {
        *v27 = 2599;
        *(_QWORD *)(v26 + 32) += 2LL;
      }
      sub_BDBD80(a1, (_BYTE *)v10);
      v28 = *(_QWORD *)a1;
      v29 = *(void **)(*(_QWORD *)a1 + 32LL);
      if ( *(_QWORD *)(*(_QWORD *)a1 + 24LL) - (_QWORD)v29 <= 0xDu )
      {
        v28 = sub_CB6200(v28, "; ModuleID = '", 14);
      }
      else
      {
        qmemcpy(v29, "; ModuleID = '", 14);
        *(_QWORD *)(v28 + 32) += 14LL;
      }
      a2 = *(_QWORD *)(v21 + 168);
      v30 = sub_CB6200(v28, a2, *(_QWORD *)(v21 + 176));
      v31 = *(_WORD **)(v30 + 32);
      v32 = v30;
      if ( *(_QWORD *)(v30 + 24) - (_QWORD)v31 <= 1u )
      {
        a2 = (__int64)"'\n";
        result = sub_CB6200(v30, "'\n", 2);
      }
      else
      {
        result = 2599;
        *v31 = 2599;
        *(_QWORD *)(v32 + 32) += 2LL;
      }
    }
  }
  else
  {
    *(_BYTE *)(a1 + 152) = 1;
  }
  v9 = v48;
  if ( v48 != v50 )
    return _libc_free(v9, a2);
  return result;
}
