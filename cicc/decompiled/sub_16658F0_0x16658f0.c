// Function: sub_16658F0
// Address: 0x16658f0
//
void __fastcall sub_16658F0(_QWORD *a1, __int64 a2)
{
  char v3; // di
  unsigned int v4; // eax
  __int64 **v5; // rdx
  __int64 v6; // r12
  __int64 v7; // rbx
  __int64 v8; // rdx
  _QWORD *v9; // rsi
  _QWORD *v10; // rax
  char v11; // dl
  _QWORD *v12; // r10
  _QWORD *v13; // rcx
  __int64 v14; // rax
  __int64 v15; // r13
  __int64 v16; // r12
  _BYTE *v17; // rax
  __int64 v18; // rax
  unsigned __int64 v19; // rdi
  __int64 v20; // r12
  _BYTE *v21; // rax
  __int64 v22; // rax
  __int64 v24; // [rsp+8h] [rbp-188h]
  _QWORD v25[2]; // [rsp+10h] [rbp-180h] BYREF
  char v26; // [rsp+20h] [rbp-170h]
  char v27; // [rsp+21h] [rbp-16Fh]
  __int64 v28; // [rsp+30h] [rbp-160h] BYREF
  _BYTE *v29; // [rsp+38h] [rbp-158h]
  _BYTE *v30; // [rsp+40h] [rbp-150h]
  __int64 v31; // [rsp+48h] [rbp-148h]
  int v32; // [rsp+50h] [rbp-140h]
  _BYTE v33[312]; // [rsp+58h] [rbp-138h] BYREF

  v3 = *(_BYTE *)(a2 + 23);
  v4 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  if ( (v3 & 0x40) != 0 )
    v5 = *(__int64 ***)(a2 - 8);
  else
    v5 = (__int64 **)(a2 - 24LL * v4);
  v6 = **v5;
  v28 = 0;
  v29 = v33;
  v30 = v33;
  v31 = 32;
  v32 = 0;
  v24 = (v4 >> 1) - 1;
  if ( v4 >> 1 == 1 )
  {
LABEL_30:
    sub_1665790(a1, a2);
    v19 = (unsigned __int64)v30;
    if ( v30 != v29 )
LABEL_26:
      _libc_free(v19);
    return;
  }
  v7 = 0;
  while ( 1 )
  {
    ++v7;
    if ( (v3 & 0x40) != 0 )
      v8 = *(_QWORD *)(a2 - 8);
    else
      v8 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
    v9 = *(_QWORD **)(v8 + 24LL * (unsigned int)(2 * v7));
    if ( v6 != *v9 )
    {
      v27 = 1;
      v25[0] = "Switch constants must all be same type as switch value!";
      v26 = 3;
      v20 = *a1;
      if ( *a1 )
      {
        sub_16E2CE0(v25, *a1);
        v21 = *(_BYTE **)(v20 + 24);
        if ( (unsigned __int64)v21 >= *(_QWORD *)(v20 + 16) )
        {
          sub_16E7DE0(v20, 10);
        }
        else
        {
          *(_QWORD *)(v20 + 24) = v21 + 1;
          *v21 = 10;
        }
        v22 = *a1;
        *((_BYTE *)a1 + 72) = 1;
        if ( v22 )
          sub_164FA80(a1, a2);
        goto LABEL_41;
      }
LABEL_40:
      *((_BYTE *)a1 + 72) = 1;
      goto LABEL_41;
    }
    v10 = v29;
    if ( v30 != v29 )
      goto LABEL_8;
    v12 = &v29[8 * HIDWORD(v31)];
    if ( v29 != (_BYTE *)v12 )
      break;
LABEL_32:
    if ( HIDWORD(v31) < (unsigned int)v31 )
    {
      ++HIDWORD(v31);
      *v12 = v9;
      ++v28;
      goto LABEL_9;
    }
LABEL_8:
    sub_16CCBA0(&v28, v9);
    if ( !v11 )
    {
      v3 = *(_BYTE *)(a2 + 23);
      goto LABEL_19;
    }
LABEL_9:
    if ( v24 == v7 )
      goto LABEL_30;
LABEL_10:
    v3 = *(_BYTE *)(a2 + 23);
  }
  v13 = 0;
  while ( v9 != (_QWORD *)*v10 )
  {
    if ( *v10 == -2 )
      v13 = v10;
    if ( v12 == ++v10 )
    {
      if ( !v13 )
        goto LABEL_32;
      *v13 = v9;
      --v32;
      ++v28;
      if ( v24 != v7 )
        goto LABEL_10;
      goto LABEL_30;
    }
  }
LABEL_19:
  if ( (v3 & 0x40) != 0 )
    v14 = *(_QWORD *)(a2 - 8);
  else
    v14 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
  v15 = *(_QWORD *)(v14 + 24LL * (unsigned int)(2 * v7));
  v27 = 1;
  v25[0] = "Duplicate integer as switch case";
  v26 = 3;
  v16 = *a1;
  if ( !*a1 )
    goto LABEL_40;
  sub_16E2CE0(v25, *a1);
  v17 = *(_BYTE **)(v16 + 24);
  if ( (unsigned __int64)v17 >= *(_QWORD *)(v16 + 16) )
  {
    sub_16E7DE0(v16, 10);
  }
  else
  {
    *(_QWORD *)(v16 + 24) = v17 + 1;
    *v17 = 10;
  }
  v18 = *a1;
  *((_BYTE *)a1 + 72) = 1;
  if ( v18 )
  {
    sub_164FA80(a1, a2);
    sub_164FA80(a1, v15);
    v19 = (unsigned __int64)v30;
    if ( v30 == v29 )
      return;
    goto LABEL_26;
  }
LABEL_41:
  v19 = (unsigned __int64)v30;
  if ( v30 != v29 )
    goto LABEL_26;
}
