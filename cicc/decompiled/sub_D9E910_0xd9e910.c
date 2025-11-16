// Function: sub_D9E910
// Address: 0xd9e910
//
__int64 __fastcall sub_D9E910(__int64 a1, __int64 a2, unsigned __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r15
  __int64 result; // rax
  __int64 v10; // r9
  int v11; // r8d
  __int64 v12; // rcx
  unsigned __int64 v13; // r8
  unsigned __int64 v14; // rdi
  void **v15; // rdx
  unsigned __int64 v16; // rsi
  int v17; // eax
  __int64 v18; // r14
  void *v19; // rax
  char *v20; // rax
  char v21; // al
  _BYTE *v22; // rdi
  __int64 v23; // rsi
  const void *v24; // r14
  unsigned __int64 v25; // rdx
  size_t v26; // rcx
  __int64 v27; // r8
  __int64 v28; // r9
  bool v29; // al
  _BYTE *v30; // rdi
  char *v31; // r14
  __int64 v32; // [rsp+0h] [rbp-110h]
  __int64 v33; // [rsp+8h] [rbp-108h]
  size_t n; // [rsp+10h] [rbp-100h]
  size_t na; // [rsp+10h] [rbp-100h]
  void **v36; // [rsp+28h] [rbp-E8h]
  unsigned __int64 v37; // [rsp+28h] [rbp-E8h]
  int v38; // [rsp+28h] [rbp-E8h]
  int v39; // [rsp+28h] [rbp-E8h]
  __int64 v40; // [rsp+30h] [rbp-E0h]
  _QWORD v41[2]; // [rsp+48h] [rbp-C8h] BYREF
  __int64 v42; // [rsp+58h] [rbp-B8h]
  bool v43; // [rsp+60h] [rbp-B0h]
  void *v44; // [rsp+70h] [rbp-A0h] BYREF
  unsigned __int64 v45; // [rsp+78h] [rbp-98h] BYREF
  __int64 v46; // [rsp+80h] [rbp-90h]
  __int64 v47; // [rsp+88h] [rbp-88h]
  bool v48; // [rsp+90h] [rbp-80h]
  __int64 v49; // [rsp+98h] [rbp-78h]
  __int64 v50; // [rsp+A0h] [rbp-70h]
  size_t v51; // [rsp+A8h] [rbp-68h]
  _BYTE *v52; // [rsp+B0h] [rbp-60h] BYREF
  __int64 v53; // [rsp+B8h] [rbp-58h]
  _BYTE v54[80]; // [rsp+C0h] [rbp-50h] BYREF

  v6 = a2;
  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0x100000000LL;
  *(_QWORD *)(a1 + 128) = a5;
  *(_BYTE *)(a1 + 136) = a4;
  *(_QWORD *)(a1 + 144) = 0;
  *(_BYTE *)(a1 + 152) = a6;
  if ( a3 > 1 )
    sub_D9E710(a1, a3, a3, a4, a5, a6);
  result = a2 + 88 * a3;
  v40 = result;
  if ( a2 != result )
  {
    do
    {
      v23 = *(_QWORD *)v6;
      v24 = *(const void **)(v6 + 40);
      v41[0] = 2;
      v41[1] = 0;
      v25 = *(unsigned int *)(v6 + 48);
      v42 = v23;
      v26 = *(_QWORD *)(v6 + 24);
      v27 = *(_QWORD *)(v6 + 16);
      v28 = *(_QWORD *)(v6 + 8);
      v29 = v23 != -8192 && v23 != 0 && v23 != -4096;
      if ( v29 )
      {
        v32 = *(_QWORD *)(v6 + 8);
        v33 = *(_QWORD *)(v6 + 16);
        n = *(_QWORD *)(v6 + 24);
        v37 = v25;
        sub_BD73F0((__int64)v41);
        v43 = 0;
        v25 = v37;
        v46 = 0;
        v26 = n;
        v47 = v42;
        v27 = v33;
        v28 = v32;
        v29 = v42 != -8192 && v42 != -4096 && v42 != 0;
        v45 = v41[0] & 6;
        if ( v29 )
        {
          sub_BD6050(&v45, v41[0] & 0xFFFFFFFFFFFFFFF8LL);
          v29 = v43;
          v28 = v32;
          v27 = v33;
          v26 = n;
          v25 = v37;
        }
      }
      else
      {
        v43 = 0;
        v45 = 2;
        v46 = 0;
        v47 = v23;
      }
      v48 = v29;
      v49 = v28;
      v10 = 8 * v25;
      v50 = v27;
      v11 = v25;
      v44 = &unk_49DE8C0;
      v51 = v26;
      v52 = v54;
      v53 = 0x400000000LL;
      if ( v25 > 4 )
      {
        na = 8 * v25;
        v38 = v25;
        sub_C8D5F0((__int64)&v52, v54, v25, 8u, v25, v10);
        v11 = v38;
        v10 = na;
        v30 = &v52[8 * (unsigned int)v53];
      }
      else
      {
        if ( !v10 )
          goto LABEL_8;
        v30 = v54;
      }
      v39 = v11;
      memcpy(v30, v24, v10);
      v10 = (unsigned int)v53;
      v11 = v39;
LABEL_8:
      LODWORD(v53) = v11 + v10;
      if ( !v43 && v42 != 0 && v42 != -4096 && v42 != -8192 )
        sub_BD60C0(v41);
      v12 = *(unsigned int *)(a1 + 8);
      v13 = *(unsigned int *)(a1 + 12);
      v14 = *(_QWORD *)a1;
      v15 = &v44;
      v16 = v12 + 1;
      v17 = *(_DWORD *)(a1 + 8);
      if ( v12 + 1 > v13 )
      {
        if ( v14 > (unsigned __int64)&v44 || (unsigned __int64)&v44 >= v14 + 112 * v12 )
        {
          sub_D9E710(a1, v16, (__int64)&v44, v12, v13, v10);
          v12 = *(unsigned int *)(a1 + 8);
          v14 = *(_QWORD *)a1;
          v15 = &v44;
          v17 = *(_DWORD *)(a1 + 8);
        }
        else
        {
          v31 = (char *)&v44 - v14;
          sub_D9E710(a1, v16, (__int64)&v44, v12, v13, v10);
          v14 = *(_QWORD *)a1;
          v12 = *(unsigned int *)(a1 + 8);
          v15 = (void **)&v31[*(_QWORD *)a1];
          v17 = *(_DWORD *)(a1 + 8);
        }
      }
      v18 = v14 + 112 * v12;
      if ( v18 )
      {
        v19 = v15[1];
        *(_QWORD *)(v18 + 16) = 0;
        *(_QWORD *)(v18 + 8) = (unsigned __int8)v19 & 6;
        v20 = (char *)v15[3];
        *(_QWORD *)(v18 + 24) = v20;
        LOBYTE(v16) = v20 + 4096 != 0;
        LOBYTE(v12) = v20 != 0;
        if ( ((v20 != 0) & (unsigned __int8)v16) != 0 && v20 != (char *)-8192LL )
        {
          v36 = v15;
          v16 = (unsigned __int64)v15[1] & 0xFFFFFFFFFFFFFFF8LL;
          sub_BD6050((unsigned __int64 *)(v18 + 8), v16);
          v15 = v36;
        }
        v21 = *((_BYTE *)v15 + 32);
        *(_QWORD *)v18 = &unk_49DE8C0;
        *(_BYTE *)(v18 + 32) = v21;
        *(_QWORD *)(v18 + 40) = v15[5];
        *(_QWORD *)(v18 + 48) = v15[6];
        *(_QWORD *)(v18 + 56) = v15[7];
        *(_QWORD *)(v18 + 64) = v18 + 80;
        *(_QWORD *)(v18 + 72) = 0x400000000LL;
        if ( *((_DWORD *)v15 + 18) )
        {
          v16 = (unsigned __int64)(v15 + 8);
          sub_D91460(v18 + 64, (char **)v15 + 8, (__int64)v15, v12, v13, v10);
        }
        v17 = *(_DWORD *)(a1 + 8);
      }
      v22 = v52;
      result = (unsigned int)(v17 + 1);
      *(_DWORD *)(a1 + 8) = result;
      if ( v22 != v54 )
        result = _libc_free(v22, v16);
      if ( !v48 )
      {
        result = v47;
        v44 = &unk_49DB368;
        if ( v47 != 0 && v47 != -4096 && v47 != -8192 )
          result = sub_BD60C0(&v45);
      }
      v6 += 88;
    }
    while ( v40 != v6 );
  }
  return result;
}
