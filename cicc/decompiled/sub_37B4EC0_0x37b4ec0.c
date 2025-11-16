// Function: sub_37B4EC0
// Address: 0x37b4ec0
//
void __fastcall sub_37B4EC0(__int64 a1, __int64 a2)
{
  __int64 (*v3)(void); // rdx
  __int64 v4; // rax
  __int64 v5; // r12
  __int64 (*v6)(); // rax
  __int64 v7; // rax
  __int64 v8; // rdi
  __int64 (*v9)(); // rdx
  __int64 v10; // rax
  unsigned __int64 v11; // r12
  volatile signed __int32 *v12; // r14
  signed __int32 v13; // eax
  signed __int32 v14; // eax
  volatile signed __int32 *v15; // r14
  signed __int32 v16; // eax
  signed __int32 v17; // eax
  __int64 v18; // rcx
  unsigned __int64 v19; // r12
  unsigned __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rcx
  unsigned __int64 v23; // rax
  _BYTE *v24; // rdx
  _BYTE *v25; // rdi
  _BYTE *v26; // rdx
  _BYTE *v27; // rdi
  _QWORD *v28; // rdi
  _QWORD *v29; // r15
  _QWORD *v30; // r12
  __int64 v31; // r14
  __int64 (*v32)(); // rcx
  int v33; // eax
  __int64 v34; // rax

  *(_BYTE *)(a1 + 12) = 0;
  *(_DWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)a1 = &unk_4A3D478;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 88) = 0;
  *(_QWORD *)(a1 + 96) = 0;
  *(_QWORD *)(a1 + 104) = 0;
  *(_QWORD *)(a1 + 112) = 0;
  *(_QWORD *)(a1 + 120) = a1;
  v3 = *(__int64 (**)(void))(**(_QWORD **)(*(_QWORD *)(a2 + 40) + 16LL) + 216LL);
  v4 = 0;
  if ( v3 != sub_2F391C0 )
    v4 = v3();
  *(_QWORD *)(a1 + 152) = v4;
  *(_QWORD *)(a1 + 160) = 0;
  *(_QWORD *)(a1 + 168) = 0;
  *(_QWORD *)(a1 + 176) = 0;
  *(_QWORD *)(a1 + 184) = 0;
  v5 = *(_QWORD *)(*(_QWORD *)(a2 + 40) + 16LL);
  *(_QWORD *)(a1 + 128) = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v5 + 200LL))(v5);
  *(_QWORD *)(a1 + 136) = *(_QWORD *)(a2 + 808);
  v6 = *(__int64 (**)())(*(_QWORD *)v5 + 128LL);
  if ( v6 == sub_2DAC790 )
  {
    *(_QWORD *)(a1 + 144) = 0;
    BUG();
  }
  v7 = ((__int64 (__fastcall *)(__int64))v6)(v5);
  *(_QWORD *)(a1 + 144) = v7;
  v8 = v7;
  v9 = *(__int64 (**)())(*(_QWORD *)v7 + 1256LL);
  v10 = 0;
  if ( v9 != sub_2FDC7B0 )
    v10 = ((__int64 (__fastcall *)(__int64, __int64))v9)(v8, v5);
  v11 = *(_QWORD *)(a1 + 160);
  *(_QWORD *)(a1 + 160) = v10;
  if ( v11 )
  {
    v12 = *(volatile signed __int32 **)(v11 + 32);
    if ( v12 )
    {
      if ( &_pthread_key_create )
      {
        v13 = _InterlockedExchangeAdd(v12 + 2, 0xFFFFFFFF);
      }
      else
      {
        v13 = *((_DWORD *)v12 + 2);
        *((_DWORD *)v12 + 2) = v13 - 1;
      }
      if ( v13 == 1 )
      {
        (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v12 + 16LL))(v12);
        if ( &_pthread_key_create )
        {
          v14 = _InterlockedExchangeAdd(v12 + 3, 0xFFFFFFFF);
        }
        else
        {
          v14 = *((_DWORD *)v12 + 3);
          *((_DWORD *)v12 + 3) = v14 - 1;
        }
        if ( v14 == 1 )
          (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v12 + 24LL))(v12);
      }
    }
    v15 = *(volatile signed __int32 **)(v11 + 16);
    if ( v15 )
    {
      if ( &_pthread_key_create )
      {
        v16 = _InterlockedExchangeAdd(v15 + 2, 0xFFFFFFFF);
      }
      else
      {
        v16 = *((_DWORD *)v15 + 2);
        *((_DWORD *)v15 + 2) = v16 - 1;
      }
      if ( v16 == 1 )
      {
        (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v15 + 16LL))(v15);
        if ( &_pthread_key_create )
        {
          v17 = _InterlockedExchangeAdd(v15 + 3, 0xFFFFFFFF);
        }
        else
        {
          v17 = *((_DWORD *)v15 + 3);
          *((_DWORD *)v15 + 3) = v17 - 1;
        }
        if ( v17 == 1 )
          (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v15 + 24LL))(v15);
      }
    }
    j_j___libc_free_0(v11);
  }
  v18 = *(_QWORD *)(a1 + 96);
  v19 = (unsigned int)((__int64)(*(_QWORD *)(*(_QWORD *)(a1 + 128) + 288LL) - *(_QWORD *)(*(_QWORD *)(a1 + 128) + 280LL)) >> 3);
  v20 = (*(_QWORD *)(a1 + 104) - v18) >> 2;
  if ( v19 > v20 )
  {
    sub_C17A60(a1 + 96, v19 - v20);
  }
  else if ( v19 < v20 )
  {
    v21 = v18 + 4 * v19;
    if ( *(_QWORD *)(a1 + 104) != v21 )
      *(_QWORD *)(a1 + 104) = v21;
  }
  v22 = *(_QWORD *)(a1 + 72);
  v23 = (*(_QWORD *)(a1 + 80) - v22) >> 2;
  if ( v19 > v23 )
  {
    sub_C17A60(a1 + 72, v19 - v23);
  }
  else if ( v19 < v23 )
  {
    v34 = v22 + 4 * v19;
    if ( *(_QWORD *)(a1 + 80) != v34 )
      *(_QWORD *)(a1 + 80) = v34;
  }
  v24 = *(_BYTE **)(a1 + 104);
  v25 = *(_BYTE **)(a1 + 96);
  if ( v25 != v24 )
    memset(v25, 0, v24 - v25);
  v26 = *(_BYTE **)(a1 + 80);
  v27 = *(_BYTE **)(a1 + 72);
  if ( v27 != v26 )
    memset(v27, 0, v26 - v27);
  v28 = *(_QWORD **)(a1 + 128);
  v29 = (_QWORD *)v28[36];
  v30 = (_QWORD *)v28[35];
  if ( v30 != v29 )
  {
    while ( 1 )
    {
      v31 = *v30;
      v32 = *(__int64 (**)())(*v28 + 360LL);
      v33 = 0;
      if ( v32 != sub_2FF5280 )
        v33 = ((__int64 (__fastcall *)(_QWORD *, _QWORD, _QWORD, __int64 (*)(), __int64 (*)()))v32)(
                v28,
                *v30,
                *(_QWORD *)(a2 + 40),
                v32,
                sub_2FF5280);
      ++v30;
      *(_DWORD *)(*(_QWORD *)(a1 + 96) + 4LL * *(unsigned __int16 *)(*(_QWORD *)v31 + 24LL)) = v33;
      if ( v29 == v30 )
        break;
      v28 = *(_QWORD **)(a1 + 128);
    }
  }
  *(_QWORD *)(a1 + 192) = 0;
}
