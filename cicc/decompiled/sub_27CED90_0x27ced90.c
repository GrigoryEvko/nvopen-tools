// Function: sub_27CED90
// Address: 0x27ced90
//
__int64 __fastcall sub_27CED90(
        unsigned __int64 a1,
        unsigned int a2,
        __int64 a3,
        __int64 a4,
        unsigned __int64 a5,
        __int64 a6)
{
  __int64 v9; // r14
  int v10; // edx
  int v11; // edx
  __int64 v12; // rax
  __int64 v13; // r13
  _BYTE *v14; // r14
  __int64 v15; // rax
  __int64 v16; // rsi
  unsigned int v17; // ecx
  _QWORD *v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rdx
  __int64 v22; // r9
  unsigned int v24; // esi
  __int64 v25; // rdi
  __int64 v26; // rax
  unsigned int v27; // ecx
  _QWORD *v28; // rdx
  __int64 v29; // r8
  unsigned __int64 v30; // r13
  int v31; // edx
  __int64 v32; // rax
  unsigned __int64 v33; // rdx
  unsigned __int64 v34; // rdi
  __int64 *v35; // rdi
  int v36; // edx
  __int64 v37; // rax
  int v38; // edx
  int v39; // r10d
  __int64 **v40; // [rsp+8h] [rbp-A8h]
  char v41; // [rsp+18h] [rbp-98h]
  __int64 v42; // [rsp+18h] [rbp-98h]
  __int64 v43; // [rsp+18h] [rbp-98h]
  __int64 v44; // [rsp+28h] [rbp-88h]
  unsigned __int64 v45[2]; // [rsp+30h] [rbp-80h] BYREF
  __int64 v46; // [rsp+40h] [rbp-70h]
  unsigned __int64 v47; // [rsp+50h] [rbp-60h] BYREF
  __int64 v48; // [rsp+58h] [rbp-58h]
  _QWORD v49[10]; // [rsp+60h] [rbp-50h] BYREF

  v9 = *(_QWORD *)(a1 + 8);
  v10 = *(unsigned __int8 *)(v9 + 8);
  if ( (unsigned int)(v10 - 17) <= 1 )
    LOBYTE(v10) = *(_BYTE *)(**(_QWORD **)(v9 + 16) + 8LL);
  if ( (_BYTE)v10 == 14 )
  {
    v35 = (__int64 *)sub_BCE3C0(*(__int64 **)v9, a2);
    v36 = *(unsigned __int8 *)(v9 + 8);
    if ( (unsigned int)(v36 - 17) > 1 )
    {
      v9 = (__int64)v35;
    }
    else
    {
      BYTE4(v44) = (_BYTE)v36 == 18;
      LODWORD(v44) = *(_DWORD *)(v9 + 32);
      v9 = sub_BCE1B0(v35, v44);
    }
  }
  v11 = *(unsigned __int16 *)(a1 + 2);
  v12 = *(_DWORD *)(a1 + 4) & 0x7FFFFFF;
  switch ( v11 )
  {
    case '2':
      v34 = *(_QWORD *)(a1 - 32 * v12);
      return sub_AD4C90(v34, (__int64 **)v9, 0);
    case '1':
      v24 = *(_DWORD *)(a3 + 24);
      if ( v24 )
      {
        v25 = *(_QWORD *)(a3 + 8);
        v26 = *(_QWORD *)(a1 - 32 * v12);
        v27 = (v24 - 1) & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
        v28 = (_QWORD *)(v25 + ((unsigned __int64)v27 << 6));
        v29 = v28[3];
        if ( v26 == v29 )
        {
LABEL_32:
          if ( v28 != (_QWORD *)(v25 + ((unsigned __int64)v24 << 6)) )
          {
            v47 = 6;
            v48 = 0;
            v49[0] = v28[7];
            v30 = v49[0];
            if ( v49[0] != 0 && v49[0] != -4096 && v49[0] != -8192 )
            {
              sub_BD6050(&v47, v28[5] & 0xFFFFFFFFFFFFFFF8LL);
              v30 = v49[0];
            }
            if ( v30 )
            {
              if ( v30 != -4096 && v30 != -8192 )
                sub_BD60C0(&v47);
              return sub_AD4C90(v30, (__int64 **)v9, 0);
            }
          }
        }
        else
        {
          v38 = 1;
          while ( v29 != -4096 )
          {
            v39 = v38 + 1;
            v27 = (v24 - 1) & (v38 + v27);
            v28 = (_QWORD *)(v25 + ((unsigned __int64)v27 << 6));
            v29 = v28[3];
            if ( v26 == v29 )
              goto LABEL_32;
            v38 = v39;
          }
        }
      }
      return sub_ADA8A0(a1, v9, 0);
    case '0':
      v34 = *(_QWORD *)(*(_QWORD *)(a1 - 32 * v12) - 32LL * (*(_DWORD *)(*(_QWORD *)(a1 - 32 * v12) + 4LL) & 0x7FFFFFF));
      return sub_AD4C90(v34, (__int64 **)v9, 0);
  }
  v13 = 0;
  v47 = (unsigned __int64)v49;
  v48 = 0x400000000LL;
  if ( !(_DWORD)v12 )
    return v13;
  v41 = 0;
  v40 = (__int64 **)v9;
  do
  {
    v14 = *(_BYTE **)(a1 + 32 * (v13 - v12));
    v15 = *(unsigned int *)(a3 + 24);
    if ( (_DWORD)v15 )
    {
      v16 = *(_QWORD *)(a3 + 8);
      v17 = (v15 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
      v18 = (_QWORD *)(v16 + ((unsigned __int64)v17 << 6));
      a6 = v18[3];
      if ( v14 == (_BYTE *)a6 )
      {
LABEL_11:
        if ( v18 != (_QWORD *)(v16 + (v15 << 6)) )
        {
          v19 = v18[7];
          v45[0] = 6;
          v45[1] = 0;
          v46 = v19;
          if ( v19 != -4096 && v19 != 0 && v19 != -8192 )
          {
            sub_BD6050(v45, v18[5] & 0xFFFFFFFFFFFFFFF8LL);
            v19 = v46;
          }
          if ( v19 )
          {
            if ( v19 != -8192 && v19 != -4096 )
            {
              v42 = v19;
              sub_BD60C0(v45);
              v19 = v42;
            }
LABEL_19:
            v20 = (unsigned int)v48;
            a5 = (unsigned int)v48 + 1LL;
            if ( a5 > HIDWORD(v48) )
            {
              v43 = v19;
              sub_C8D5F0((__int64)&v47, v49, (unsigned int)v48 + 1LL, 8u, a5, a6);
              v20 = (unsigned int)v48;
              v19 = v43;
            }
            *(_QWORD *)(v47 + 8 * v20) = v19;
            LODWORD(v48) = v48 + 1;
            v41 = 1;
            goto LABEL_22;
          }
        }
      }
      else
      {
        v31 = 1;
        while ( a6 != -4096 )
        {
          a5 = (unsigned int)(v31 + 1);
          v17 = (v15 - 1) & (v31 + v17);
          v18 = (_QWORD *)(v16 + ((unsigned __int64)v17 << 6));
          a6 = v18[3];
          if ( v14 == (_BYTE *)a6 )
            goto LABEL_11;
          v31 = a5;
        }
      }
    }
    if ( (*(_WORD *)(a1 + 2) != 34 || !(_DWORD)v13) && *v14 == 5 )
    {
      v19 = sub_27CED90(v14, a2, a3);
      if ( v19 )
        goto LABEL_19;
    }
    v32 = (unsigned int)v48;
    v33 = (unsigned int)v48 + 1LL;
    if ( v33 > HIDWORD(v48) )
    {
      sub_C8D5F0((__int64)&v47, v49, v33, 8u, a5, a6);
      v32 = (unsigned int)v48;
    }
    *(_QWORD *)(v47 + 8 * v32) = v14;
    LODWORD(v48) = v48 + 1;
LABEL_22:
    ++v13;
    v12 = *(_DWORD *)(a1 + 4) & 0x7FFFFFF;
  }
  while ( (unsigned int)v12 > (unsigned int)v13 );
  if ( v41 )
  {
    if ( *(_WORD *)(a1 + 2) == 34 )
    {
      v37 = sub_BB5290(a1);
      v21 = (unsigned int)v48;
      v22 = v37;
    }
    else
    {
      v21 = (unsigned int)v48;
      v22 = 0;
    }
    v13 = sub_ADABF0(a1, v47, v21, v40, 0, v22);
  }
  else
  {
    v13 = 0;
  }
  if ( (_QWORD *)v47 != v49 )
    _libc_free(v47);
  return v13;
}
