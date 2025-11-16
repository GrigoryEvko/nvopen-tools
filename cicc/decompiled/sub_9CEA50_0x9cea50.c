// Function: sub_9CEA50
// Address: 0x9cea50
//
__int64 __fastcall sub_9CEA50(__int64 a1, __int64 a2, char a3, __int64 a4)
{
  char v6; // al
  __int64 v7; // rax
  __int64 v8; // rdi
  char v10; // al
  int v11; // ecx
  unsigned __int64 v12; // rax
  char v13; // al
  char v14; // al
  char v15; // dl
  char v16; // dl
  char v17; // al
  __int64 v18; // rax
  unsigned int v19; // edx
  __int64 v20; // r15
  __int64 v21; // rcx
  __int64 v22; // rax
  __int64 v23; // rbx
  volatile signed __int32 *v24; // r14
  signed __int32 v25; // edx
  void (*v26)(); // rdx
  signed __int32 v27; // eax
  __int64 (__fastcall *v28)(__int64); // rdx
  __int64 v29; // r14
  _QWORD *v30; // r14
  __int64 v31; // r15
  __int64 i; // rbx
  volatile signed __int32 *v33; // r13
  signed __int32 v34; // edx
  void (*v35)(); // rdx
  signed __int32 v36; // eax
  __int64 (__fastcall *v37)(__int64); // rdx
  __int64 v38; // rdi
  char v39; // dl
  __int64 v40; // rax
  __int64 v41; // [rsp+10h] [rbp-70h]
  __int64 v42; // [rsp+10h] [rbp-70h]
  __int64 v43; // [rsp+18h] [rbp-68h]
  __int64 v44; // [rsp+28h] [rbp-58h] BYREF
  __int64 v45; // [rsp+30h] [rbp-50h] BYREF
  char v46; // [rsp+38h] [rbp-48h]
  __int64 v47; // [rsp+40h] [rbp-40h] BYREF
  char v48; // [rsp+48h] [rbp-38h]

  while ( 1 )
  {
    if ( !*(_DWORD *)(a2 + 32) && *(_QWORD *)(a2 + 8) <= *(_QWORD *)(a2 + 16) )
      goto LABEL_14;
    sub_9C66D0((__int64)&v47, a2, *(unsigned int *)(a2 + 36), a4);
    v6 = v48 & 1 | v46 & 0xFE | 2;
    v46 = v6;
    if ( (v48 & 1) != 0 )
    {
      v7 = v47;
      *(_BYTE *)(a1 + 8) |= 3u;
      v45 = 0;
      *(_QWORD *)a1 = v7 & 0xFFFFFFFFFFFFFFFELL;
LABEL_5:
      v8 = v45;
      if ( !v45 )
        return a1;
      goto LABEL_6;
    }
    v10 = v6 & 0xFD;
    v46 = v10;
    LODWORD(v45) = v47;
    v11 = v47;
    if ( !(_DWORD)v47 )
    {
      if ( (a3 & 1) != 0 )
      {
        v15 = *(_BYTE *)(a1 + 8);
        *(_DWORD *)a1 = 1;
        *(_DWORD *)(a1 + 4) = 0;
        *(_BYTE *)(a1 + 8) = v15 & 0xFC | 2;
      }
      else
      {
        v18 = *(unsigned int *)(a2 + 72);
        if ( !(_DWORD)v18 )
        {
LABEL_14:
          v13 = *(_BYTE *)(a1 + 8);
          *(_DWORD *)a1 = 0;
          *(_DWORD *)(a1 + 4) = 0;
          *(_BYTE *)(a1 + 8) = v13 & 0xFC | 2;
          return a1;
        }
        v19 = *(_DWORD *)(a2 + 32);
        if ( v19 > 0x1F )
        {
          *(_DWORD *)(a2 + 32) = 32;
          *(_QWORD *)(a2 + 24) >>= (unsigned __int8)v19 - 32;
        }
        else
        {
          *(_DWORD *)(a2 + 32) = 0;
        }
        v20 = *(_QWORD *)(a2 + 40);
        v21 = *(_QWORD *)(a2 + 48);
        v22 = *(_QWORD *)(a2 + 64) + 32 * v18 - 32;
        v43 = *(_QWORD *)(a2 + 56);
        v23 = v20;
        *(_DWORD *)(a2 + 36) = *(_DWORD *)v22;
        *(_QWORD *)(a2 + 40) = *(_QWORD *)(v22 + 8);
        *(_QWORD *)(a2 + 48) = *(_QWORD *)(v22 + 16);
        *(_QWORD *)(a2 + 56) = *(_QWORD *)(v22 + 24);
        *(_QWORD *)(v22 + 8) = 0;
        *(_QWORD *)(v22 + 16) = 0;
        for ( *(_QWORD *)(v22 + 24) = 0; v21 != v23; v23 += 16 )
        {
          v24 = *(volatile signed __int32 **)(v23 + 8);
          if ( v24 )
          {
            if ( &_pthread_key_create )
            {
              v25 = _InterlockedExchangeAdd(v24 + 2, 0xFFFFFFFF);
            }
            else
            {
              v25 = *((_DWORD *)v24 + 2);
              *((_DWORD *)v24 + 2) = v25 - 1;
            }
            if ( v25 == 1 )
            {
              v26 = *(void (**)())(*(_QWORD *)v24 + 16LL);
              if ( v26 != nullsub_25 )
              {
                v42 = v21;
                ((void (__fastcall *)(volatile signed __int32 *))v26)(v24);
                v21 = v42;
              }
              if ( &_pthread_key_create )
              {
                v27 = _InterlockedExchangeAdd(v24 + 3, 0xFFFFFFFF);
              }
              else
              {
                v27 = *((_DWORD *)v24 + 3);
                *((_DWORD *)v24 + 3) = v27 - 1;
              }
              if ( v27 == 1 )
              {
                v41 = v21;
                v28 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v24 + 24LL);
                if ( v28 == sub_9C26E0 )
                  (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v24 + 8LL))(v24);
                else
                  v28((__int64)v24);
                v21 = v41;
              }
            }
          }
        }
        if ( v20 )
          j_j___libc_free_0(v20, v43 - v20);
        v29 = (unsigned int)(*(_DWORD *)(a2 + 72) - 1);
        *(_DWORD *)(a2 + 72) = v29;
        v30 = (_QWORD *)(*(_QWORD *)(a2 + 64) + 32 * v29);
        v31 = v30[2];
        for ( i = v30[1]; v31 != i; i += 16 )
        {
          v33 = *(volatile signed __int32 **)(i + 8);
          if ( v33 )
          {
            if ( &_pthread_key_create )
            {
              v34 = _InterlockedExchangeAdd(v33 + 2, 0xFFFFFFFF);
            }
            else
            {
              v34 = *((_DWORD *)v33 + 2);
              *((_DWORD *)v33 + 2) = v34 - 1;
            }
            if ( v34 == 1 )
            {
              v35 = *(void (**)())(*(_QWORD *)v33 + 16LL);
              if ( v35 != nullsub_25 )
                ((void (__fastcall *)(volatile signed __int32 *))v35)(v33);
              if ( &_pthread_key_create )
              {
                v36 = _InterlockedExchangeAdd(v33 + 3, 0xFFFFFFFF);
              }
              else
              {
                v36 = *((_DWORD *)v33 + 3);
                *((_DWORD *)v33 + 3) = v36 - 1;
              }
              if ( v36 == 1 )
              {
                v37 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v33 + 24LL);
                if ( v37 == sub_9C26E0 )
                  (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v33 + 8LL))(v33);
                else
                  v37((__int64)v33);
              }
            }
          }
        }
        v38 = v30[1];
        if ( v38 )
          j_j___libc_free_0(v38, v30[3] - v38);
        v39 = *(_BYTE *)(a1 + 8);
        v10 = v46;
        *(_DWORD *)a1 = 1;
        *(_DWORD *)(a1 + 4) = 0;
        *(_BYTE *)(a1 + 8) = v39 & 0xFC | 2;
        if ( (v10 & 2) != 0 )
          sub_9CE230(&v45);
      }
      if ( (v10 & 1) == 0 )
        return a1;
      goto LABEL_5;
    }
    if ( (_DWORD)v47 == 1 )
      break;
    if ( (_DWORD)v47 != 2 || (a3 & 2) != 0 )
    {
      v14 = *(_BYTE *)(a1 + 8);
      *(_DWORD *)a1 = 3;
      *(_DWORD *)(a1 + 4) = v11;
      *(_BYTE *)(a1 + 8) = v14 & 0xFC | 2;
      return a1;
    }
    sub_A4D380(&v47, a2, v47, (unsigned int)v47);
    v12 = v47 & 0xFFFFFFFFFFFFFFFELL;
    if ( (v47 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    {
      *(_BYTE *)(a1 + 8) |= 3u;
      *(_QWORD *)a1 = v12;
      return a1;
    }
  }
  sub_9CE2D0((__int64)&v47, a2, 8, 1);
  v16 = v48 & 1;
  v48 = (2 * (v48 & 1)) | v48 & 0xFD;
  if ( !v16 )
  {
    v17 = *(_BYTE *)(a1 + 8);
    *(_DWORD *)a1 = 2;
    *(_BYTE *)(a1 + 8) = v17 & 0xFC | 2;
    *(_DWORD *)(a1 + 4) = v47;
    return a1;
  }
  sub_9C8CD0(&v44, &v47);
  v40 = v44;
  *(_BYTE *)(a1 + 8) |= 3u;
  *(_QWORD *)a1 = v40 & 0xFFFFFFFFFFFFFFFELL;
  if ( (v48 & 2) != 0 )
    sub_9CE230(&v47);
  if ( (v48 & 1) != 0 )
  {
    v8 = v47;
    if ( v47 )
LABEL_6:
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v8 + 8LL))(v8);
  }
  return a1;
}
