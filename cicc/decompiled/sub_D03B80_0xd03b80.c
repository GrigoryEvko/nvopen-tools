// Function: sub_D03B80
// Address: 0xd03b80
//
__int64 __fastcall sub_D03B80(__int64 a1, __int64 *a2, __int64 a3, char a4)
{
  unsigned int v4; // r14d
  _QWORD *v5; // rdx
  unsigned int v7; // ebx
  __int64 v8; // rax
  unsigned int v9; // eax
  __int64 v10; // rsi
  unsigned __int8 *v11; // rdi
  unsigned __int8 **v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  unsigned __int8 *v16; // r15
  unsigned __int8 **v17; // rax
  _QWORD *v18; // rdi
  unsigned int v19; // eax
  __int64 v20; // rdx
  char v22; // dl
  unsigned __int8 v23; // al
  __int64 v24; // rax
  __int64 v25; // rax
  unsigned __int8 *v26; // r10
  __int64 v27; // r8
  __int64 v28; // rdx
  __int64 v29; // rax
  _QWORD *v30; // rcx
  __int64 v31; // rax
  __int64 v32; // r10
  unsigned __int64 v33; // rdx
  __int64 v34; // rax
  __int64 v35; // r15
  int v36; // [rsp+0h] [rbp-F0h]
  __int64 v37; // [rsp+8h] [rbp-E8h]
  __int64 v38; // [rsp+10h] [rbp-E0h]
  unsigned __int8 *v39; // [rsp+10h] [rbp-E0h]
  __int64 v40; // [rsp+20h] [rbp-D0h]
  _QWORD *v42; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v43; // [rsp+38h] [rbp-B8h]
  _QWORD v44[22]; // [rsp+40h] [rbp-B0h] BYREF

  v4 = 0;
  v5 = v44;
  v7 = 8;
  v8 = *a2;
  v42 = v44;
  v44[0] = v8;
  v43 = 0x1000000001LL;
  v40 = a1 + 40;
  v9 = 1;
  while ( 1 )
  {
    v10 = 6;
    v11 = (unsigned __int8 *)v5[v9 - 1];
    LODWORD(v43) = v9 - 1;
    v16 = sub_98ACB0(v11, 6u);
    if ( *(_BYTE *)(a1 + 68) )
    {
      v17 = *(unsigned __int8 ***)(a1 + 48);
      v13 = *(unsigned int *)(a1 + 60);
      v12 = &v17[v13];
      if ( v17 != v12 )
      {
        while ( v16 != *v17 )
        {
          if ( v12 == ++v17 )
            goto LABEL_20;
        }
        goto LABEL_7;
      }
LABEL_20:
      if ( (unsigned int)v13 < *(_DWORD *)(a1 + 56) )
        break;
    }
    v10 = (__int64)v16;
    sub_C8CC70(v40, (__int64)v16, (__int64)v12, v13, v14, v15);
    if ( v22 )
      goto LABEL_22;
LABEL_7:
    v9 = v43;
    v18 = v42;
LABEL_8:
    v5 = v18;
    if ( !v9 )
      goto LABEL_11;
    if ( !--v7 )
      goto LABEL_10;
  }
  *(_DWORD *)(a1 + 60) = v13 + 1;
  *v12 = v16;
  ++*(_QWORD *)(a1 + 40);
LABEL_22:
  v23 = *v16;
  if ( a4 && v23 == 60 )
    goto LABEL_7;
  if ( v23 == 22 )
  {
    if ( (unsigned __int8)sub_B2D700((__int64)v16) && (unsigned __int8)sub_B2BD80((__int64)v16) )
    {
      v9 = v43;
      v18 = v42;
      v4 = 1;
      goto LABEL_8;
    }
    v23 = *v16;
  }
  switch ( v23 )
  {
    case 3u:
      if ( (v16[80] & 1) == 0 )
        break;
      goto LABEL_7;
    case 0x56u:
      v31 = (unsigned int)v43;
      v32 = *((_QWORD *)v16 - 8);
      v33 = (unsigned int)v43 + 1LL;
      if ( v33 > HIDWORD(v43) )
      {
        v10 = (__int64)v44;
        v38 = *((_QWORD *)v16 - 8);
        sub_C8D5F0((__int64)&v42, v44, v33, 8u, v14, v15);
        v31 = (unsigned int)v43;
        v32 = v38;
      }
      v42[v31] = v32;
      LODWORD(v43) = v43 + 1;
      v34 = (unsigned int)v43;
      v35 = *((_QWORD *)v16 - 4);
      if ( (unsigned __int64)(unsigned int)v43 + 1 > HIDWORD(v43) )
      {
        v10 = (__int64)v44;
        sub_C8D5F0((__int64)&v42, v44, (unsigned int)v43 + 1LL, 8u, v14, v15);
        v34 = (unsigned int)v43;
      }
      v42[v34] = v35;
      v18 = v42;
      v9 = v43 + 1;
      LODWORD(v43) = v43 + 1;
      goto LABEL_8;
    case 0x54u:
      v24 = *((_DWORD *)v16 + 1) & 0x7FFFFFF;
      if ( (unsigned int)v24 <= v7 )
      {
        v25 = 32 * v24;
        if ( (v16[7] & 0x40) != 0 )
        {
          v26 = (unsigned __int8 *)*((_QWORD *)v16 - 1);
          v27 = (__int64)&v26[v25];
        }
        else
        {
          v27 = (__int64)v16;
          v26 = &v16[-v25];
        }
        v28 = (unsigned int)v43;
        v29 = v25 >> 5;
        if ( (unsigned __int64)(unsigned int)v43 + v29 > HIDWORD(v43) )
        {
          v10 = (__int64)v44;
          v36 = v29;
          v37 = v27;
          v39 = v26;
          sub_C8D5F0((__int64)&v42, v44, (unsigned int)v43 + v29, 8u, v27, v15);
          v28 = (unsigned int)v43;
          LODWORD(v29) = v36;
          v27 = v37;
          v26 = v39;
        }
        v18 = v42;
        v30 = &v42[v28];
        if ( v26 != (unsigned __int8 *)v27 )
        {
          do
          {
            if ( v30 )
              *v30 = *(_QWORD *)v26;
            v26 += 32;
            ++v30;
          }
          while ( (unsigned __int8 *)v27 != v26 );
          LODWORD(v28) = v43;
          v18 = v42;
        }
        LODWORD(v43) = v29 + v28;
        v9 = v29 + v28;
        goto LABEL_8;
      }
      break;
  }
  v18 = v42;
LABEL_10:
  v4 = 3;
LABEL_11:
  if ( v18 != v44 )
    _libc_free(v18, v10);
  ++*(_QWORD *)(a1 + 40);
  if ( *(_BYTE *)(a1 + 68) )
  {
LABEL_18:
    *(_QWORD *)(a1 + 60) = 0;
  }
  else
  {
    v19 = 4 * (*(_DWORD *)(a1 + 60) - *(_DWORD *)(a1 + 64));
    v20 = *(unsigned int *)(a1 + 56);
    if ( v19 < 0x20 )
      v19 = 32;
    if ( (unsigned int)v20 <= v19 )
    {
      memset(*(void **)(a1 + 48), -1, 8 * v20);
      goto LABEL_18;
    }
    sub_C8C990(v40, v10);
  }
  return v4;
}
