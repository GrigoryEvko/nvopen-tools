// Function: sub_37011E0
// Address: 0x37011e0
//
unsigned __int64 *__fastcall sub_37011E0(unsigned __int64 *a1, _QWORD *a2, unsigned int *a3, __int64 *a4)
{
  __int64 v7; // rsi
  __int64 v8; // r15
  __int64 v9; // rsi
  unsigned __int64 v10; // rax
  int v11; // r8d
  unsigned int v12; // eax
  unsigned __int32 v13; // edx
  __int64 v14; // rsi
  unsigned int v15; // ebx
  __int64 (*v16)(void); // rax
  char v19; // dl
  char v20; // r14
  __int64 v21; // rdi
  __int64 v22; // rcx
  unsigned __int64 *v23; // rsi
  _DWORD *v24; // rdi
  int v25; // r8d
  unsigned __int32 v26; // eax
  __int64 v27; // rcx
  _DWORD *v28; // rsi
  _DWORD *v29; // rdi
  unsigned __int64 v30; // rax
  unsigned __int64 v31; // [rsp+8h] [rbp-F8h]
  unsigned __int64 v32; // [rsp+10h] [rbp-F0h]
  char v33; // [rsp+10h] [rbp-F0h]
  char **v34; // [rsp+18h] [rbp-E8h]
  unsigned __int64 v35[2]; // [rsp+20h] [rbp-E0h] BYREF
  __int64 v36; // [rsp+30h] [rbp-D0h] BYREF
  char *v37; // [rsp+40h] [rbp-C0h] BYREF
  unsigned __int64 v38; // [rsp+48h] [rbp-B8h]
  char *v39; // [rsp+50h] [rbp-B0h]
  __int16 v40; // [rsp+60h] [rbp-A0h]
  unsigned __int64 v41[2]; // [rsp+70h] [rbp-90h] BYREF
  unsigned __int64 *v42; // [rsp+80h] [rbp-80h]
  char v43; // [rsp+90h] [rbp-70h]
  char v44; // [rsp+91h] [rbp-6Fh]
  _QWORD v45[4]; // [rsp+A0h] [rbp-60h] BYREF
  unsigned __int8 v46; // [rsp+C0h] [rbp-40h]

  v7 = a2[7];
  v8 = a2[5];
  if ( !v7 )
  {
    v14 = a2[6];
    if ( v14 && !v8 )
    {
      v15 = *a3;
      v16 = *(__int64 (**)(void))(**(_QWORD **)(v14 + 24) + 16LL);
      if ( v16 != sub_3700C70 )
      {
        v14 = a2[6];
        v25 = v16();
        v26 = _byteswap_ulong(v15);
        if ( v25 != 1 )
          v15 = v26;
      }
      LODWORD(v41[0]) = v15;
      sub_3719260(v45, v14, v41, 4);
      v10 = v45[0] & 0xFFFFFFFFFFFFFFFELL;
      if ( (v45[0] & 0xFFFFFFFFFFFFFFFELL) == 0 )
        goto LABEL_7;
      goto LABEL_12;
    }
LABEL_3:
    v9 = a2[5];
    v45[0] = 0;
    v45[1] = 0;
    sub_1254950(v41, v9, (__int64)v45, 4u);
    v10 = v41[0] & 0xFFFFFFFFFFFFFFFELL;
    if ( (v41[0] & 0xFFFFFFFFFFFFFFFELL) == 0 )
    {
      v11 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(v8 + 24) + 16LL))(*(_QWORD *)(v8 + 24));
      v12 = *(_DWORD *)v45[0];
      v13 = _byteswap_ulong(*(_DWORD *)v45[0]);
      if ( v11 != 1 )
        v12 = v13;
      *a3 = v12;
      goto LABEL_7;
    }
LABEL_12:
    *a1 = v10 | 1;
    return a1;
  }
  if ( v8 || a2[6] )
    goto LABEL_3;
  (*(void (__fastcall **)(unsigned __int64 *, __int64, _QWORD))(*(_QWORD *)v7 + 48LL))(v35, v7, *a3);
  if ( v35[1] )
  {
    v19 = *((_BYTE *)a4 + 32);
    if ( v19 )
    {
      if ( v19 == 1 )
      {
        v40 = 259;
        v19 = 3;
        v37 = ": ";
        v31 = v38;
        v34 = (char **)": ";
      }
      else
      {
        if ( *((_BYTE *)a4 + 33) == 1 )
        {
          v30 = a4[1];
          a4 = (__int64 *)*a4;
          v32 = v30;
        }
        else
        {
          v19 = 2;
        }
        v37 = (char *)a4;
        HIBYTE(v40) = 3;
        v38 = v32;
        v39 = ": ";
        v34 = &v37;
        LOBYTE(v40) = v19;
        v19 = 2;
      }
      v42 = v35;
      v20 = 4;
      v41[0] = (unsigned __int64)v34;
      v41[1] = v31;
    }
    else
    {
      v20 = 1;
      v40 = 256;
    }
    v21 = a2[7];
    if ( !v21 || a2[5] || a2[6] )
      goto LABEL_33;
    v33 = v19;
    if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v21 + 40LL))(v21) )
      goto LABEL_30;
    v42 = v35;
    v22 = 10;
    v43 = v33;
    v23 = v41;
    v24 = v45;
    v41[0] = (unsigned __int64)v34;
    v44 = v20;
    while ( v22 )
    {
      *v24 = *(_DWORD *)v23;
      v23 = (unsigned __int64 *)((char *)v23 + 4);
      ++v24;
      --v22;
    }
    if ( (unsigned __int8)v33 <= 1u )
      goto LABEL_30;
    goto LABEL_49;
  }
  v21 = a2[7];
  if ( !v21 || a2[5] || a2[6] )
    goto LABEL_33;
  if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v21 + 40LL))(v21) )
  {
    v27 = 10;
    v28 = a4;
    v29 = v45;
    while ( v27 )
    {
      *v29++ = *v28++;
      --v27;
    }
    if ( v46 > 1u )
LABEL_49:
      (*(void (__fastcall **)(_QWORD, _QWORD *))(*(_QWORD *)a2[7] + 24LL))(a2[7], v45);
  }
LABEL_30:
  v21 = a2[7];
LABEL_33:
  (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v21 + 8LL))(v21, *a3, 4);
  if ( a2[7] && !a2[5] && !a2[6] )
    a2[8] += 4LL;
  if ( (__int64 *)v35[0] != &v36 )
    j_j___libc_free_0(v35[0]);
LABEL_7:
  *a1 = 1;
  return a1;
}
