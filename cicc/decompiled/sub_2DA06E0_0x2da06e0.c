// Function: sub_2DA06E0
// Address: 0x2da06e0
//
void __fastcall sub_2DA06E0(_QWORD *a1)
{
  _QWORD *v1; // r12
  _QWORD *v2; // r14
  _QWORD *v3; // r13
  unsigned __int64 v4; // rbx
  unsigned __int64 v5; // rax
  volatile signed __int32 *v6; // rdi
  int v7; // edx
  volatile signed __int32 *v8; // rdi
  _QWORD *v9; // rbx
  signed __int32 v10; // edx
  volatile signed __int32 *v11; // r13
  _QWORD *v12; // rbx
  signed __int32 v13; // edx
  signed __int32 v14; // eax
  signed __int32 v15; // eax
  volatile signed __int32 *v16; // r13
  _QWORD *v17; // rbx
  signed __int32 v18; // edx
  signed __int32 v19; // eax
  volatile signed __int32 *v20; // r12
  _QWORD *v21; // rbx
  signed __int32 v22; // edx
  signed __int32 v23; // eax
  volatile signed __int32 *v24; // r13
  _QWORD *v25; // rbx
  signed __int32 v26; // edx
  signed __int32 v27; // eax
  volatile signed __int32 *v28; // r12
  _QWORD *v29; // rbx
  signed __int32 v30; // edx
  signed __int32 v31; // eax
  volatile signed __int32 *v32; // r12
  _QWORD *v33; // rbx
  signed __int32 v34; // edx
  signed __int32 v35; // eax
  signed __int32 v36; // eax
  volatile signed __int32 *v37; // r12
  _QWORD *v38; // rbx
  signed __int32 v39; // edx
  signed __int32 v40; // eax
  unsigned __int64 v41; // [rsp+8h] [rbp-68h]
  _QWORD *v42; // [rsp+10h] [rbp-60h]
  _QWORD *v43; // [rsp+18h] [rbp-58h]
  _QWORD *v44; // [rsp+20h] [rbp-50h]
  _QWORD *v45; // [rsp+28h] [rbp-48h]
  _QWORD *v46; // [rsp+30h] [rbp-40h]

  v42 = a1;
  if ( a1 )
  {
    while ( 1 )
    {
      v45 = (_QWORD *)v42[3];
      if ( v45 )
        break;
LABEL_104:
      v37 = (volatile signed __int32 *)v42[6];
      v38 = (_QWORD *)v42[2];
      if ( v37 )
      {
        if ( &_pthread_key_create )
        {
          v39 = _InterlockedExchangeAdd(v37 + 2, 0xFFFFFFFF);
        }
        else
        {
          v39 = *((_DWORD *)v37 + 2);
          *((_DWORD *)v37 + 2) = v39 - 1;
        }
        if ( v39 == 1 )
        {
          (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v37 + 16LL))(v37);
          if ( &_pthread_key_create )
          {
            v40 = _InterlockedExchangeAdd(v37 + 3, 0xFFFFFFFF);
          }
          else
          {
            v40 = *((_DWORD *)v37 + 3);
            *((_DWORD *)v37 + 3) = v40 - 1;
          }
          if ( v40 == 1 )
            (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v37 + 24LL))(v37);
        }
      }
      j_j___libc_free_0((unsigned __int64)v42);
      if ( !v38 )
        return;
      v42 = v38;
    }
    while ( 1 )
    {
      v44 = (_QWORD *)v45[3];
      if ( v44 )
        break;
LABEL_86:
      v32 = (volatile signed __int32 *)v45[6];
      v33 = (_QWORD *)v45[2];
      if ( v32 )
      {
        if ( &_pthread_key_create )
        {
          v34 = _InterlockedExchangeAdd(v32 + 2, 0xFFFFFFFF);
        }
        else
        {
          v34 = *((_DWORD *)v32 + 2);
          *((_DWORD *)v32 + 2) = v34 - 1;
        }
        if ( v34 == 1 )
        {
          (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v32 + 16LL))(v32);
          if ( &_pthread_key_create )
          {
            v36 = _InterlockedExchangeAdd(v32 + 3, 0xFFFFFFFF);
          }
          else
          {
            v36 = *((_DWORD *)v32 + 3);
            *((_DWORD *)v32 + 3) = v36 - 1;
          }
          if ( v36 == 1 )
            (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v32 + 24LL))(v32);
        }
      }
      j_j___libc_free_0((unsigned __int64)v45);
      if ( !v33 )
        goto LABEL_104;
      v45 = v33;
    }
    while ( 1 )
    {
      v43 = (_QWORD *)v44[3];
      if ( v43 )
        break;
LABEL_73:
      v28 = (volatile signed __int32 *)v44[6];
      v29 = (_QWORD *)v44[2];
      if ( v28 )
      {
        if ( &_pthread_key_create )
        {
          v30 = _InterlockedExchangeAdd(v28 + 2, 0xFFFFFFFF);
        }
        else
        {
          v30 = *((_DWORD *)v28 + 2);
          *((_DWORD *)v28 + 2) = v30 - 1;
        }
        if ( v30 == 1 )
        {
          (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v28 + 16LL))(v28);
          if ( &_pthread_key_create )
          {
            v35 = _InterlockedExchangeAdd(v28 + 3, 0xFFFFFFFF);
          }
          else
          {
            v35 = *((_DWORD *)v28 + 3);
            *((_DWORD *)v28 + 3) = v35 - 1;
          }
          if ( v35 == 1 )
            (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v28 + 24LL))(v28);
        }
      }
      j_j___libc_free_0((unsigned __int64)v44);
      if ( !v29 )
        goto LABEL_86;
      v44 = v29;
    }
    while ( 1 )
    {
      v1 = (_QWORD *)v43[3];
      if ( v1 )
        break;
LABEL_48:
      v20 = (volatile signed __int32 *)v43[6];
      v21 = (_QWORD *)v43[2];
      if ( v20 )
      {
        if ( &_pthread_key_create )
        {
          v22 = _InterlockedExchangeAdd(v20 + 2, 0xFFFFFFFF);
        }
        else
        {
          v22 = *((_DWORD *)v20 + 2);
          *((_DWORD *)v20 + 2) = v22 - 1;
        }
        if ( v22 == 1 )
        {
          (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v20 + 16LL))(v20);
          if ( &_pthread_key_create )
          {
            v31 = _InterlockedExchangeAdd(v20 + 3, 0xFFFFFFFF);
          }
          else
          {
            v31 = *((_DWORD *)v20 + 3);
            *((_DWORD *)v20 + 3) = v31 - 1;
          }
          if ( v31 == 1 )
            (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v20 + 24LL))(v20);
        }
      }
      j_j___libc_free_0((unsigned __int64)v43);
      if ( !v21 )
        goto LABEL_73;
      v43 = v21;
    }
    while ( 1 )
    {
      v2 = (_QWORD *)v1[3];
      if ( v2 )
        break;
LABEL_20:
      v11 = (volatile signed __int32 *)v1[6];
      v12 = (_QWORD *)v1[2];
      if ( v11 )
      {
        if ( &_pthread_key_create )
        {
          v13 = _InterlockedExchangeAdd(v11 + 2, 0xFFFFFFFF);
        }
        else
        {
          v13 = *((_DWORD *)v11 + 2);
          *((_DWORD *)v11 + 2) = v13 - 1;
        }
        if ( v13 == 1 )
        {
          (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v11 + 16LL))(v11);
          if ( &_pthread_key_create )
          {
            v23 = _InterlockedExchangeAdd(v11 + 3, 0xFFFFFFFF);
          }
          else
          {
            v23 = *((_DWORD *)v11 + 3);
            *((_DWORD *)v11 + 3) = v23 - 1;
          }
          if ( v23 == 1 )
            (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v11 + 24LL))(v11);
        }
      }
      j_j___libc_free_0((unsigned __int64)v1);
      if ( !v12 )
        goto LABEL_48;
      v1 = v12;
    }
    while ( 1 )
    {
      v46 = (_QWORD *)v2[3];
      if ( v46 )
        break;
LABEL_37:
      v16 = (volatile signed __int32 *)v2[6];
      v17 = (_QWORD *)v2[2];
      if ( v16 )
      {
        if ( &_pthread_key_create )
        {
          v18 = _InterlockedExchangeAdd(v16 + 2, 0xFFFFFFFF);
        }
        else
        {
          v18 = *((_DWORD *)v16 + 2);
          *((_DWORD *)v16 + 2) = v18 - 1;
        }
        if ( v18 == 1 )
        {
          (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v16 + 16LL))(v16);
          if ( &_pthread_key_create )
          {
            v19 = _InterlockedExchangeAdd(v16 + 3, 0xFFFFFFFF);
          }
          else
          {
            v19 = *((_DWORD *)v16 + 3);
            *((_DWORD *)v16 + 3) = v19 - 1;
          }
          if ( v19 == 1 )
            (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v16 + 24LL))(v16);
        }
      }
      j_j___libc_free_0((unsigned __int64)v2);
      if ( !v17 )
        goto LABEL_20;
      v2 = v17;
    }
    while ( 1 )
    {
      v3 = (_QWORD *)v46[3];
      if ( v3 )
        break;
LABEL_61:
      v24 = (volatile signed __int32 *)v46[6];
      v25 = (_QWORD *)v46[2];
      if ( v24 )
      {
        if ( &_pthread_key_create )
        {
          v26 = _InterlockedExchangeAdd(v24 + 2, 0xFFFFFFFF);
        }
        else
        {
          v26 = *((_DWORD *)v24 + 2);
          *((_DWORD *)v24 + 2) = v26 - 1;
        }
        if ( v26 == 1 )
        {
          (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v24 + 16LL))(v24);
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
            (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v24 + 24LL))(v24);
        }
      }
      j_j___libc_free_0((unsigned __int64)v46);
      if ( !v25 )
        goto LABEL_37;
      v46 = v25;
    }
    while ( 1 )
    {
      v4 = v3[3];
      if ( v4 )
        break;
LABEL_14:
      v8 = (volatile signed __int32 *)v3[6];
      v9 = (_QWORD *)v3[2];
      if ( v8 )
      {
        if ( &_pthread_key_create )
        {
          v10 = _InterlockedExchangeAdd(v8 + 2, 0xFFFFFFFF);
        }
        else
        {
          v10 = *((_DWORD *)v8 + 2);
          *((_DWORD *)v8 + 2) = v10 - 1;
        }
        if ( v10 == 1 )
        {
          (*(void (**)(void))(*(_QWORD *)v8 + 16LL))();
          if ( &_pthread_key_create )
          {
            v15 = _InterlockedExchangeAdd(v8 + 3, 0xFFFFFFFF);
          }
          else
          {
            v15 = *((_DWORD *)v8 + 3);
            *((_DWORD *)v8 + 3) = v15 - 1;
          }
          if ( v15 == 1 )
            (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v8 + 24LL))(v8);
        }
      }
      j_j___libc_free_0((unsigned __int64)v3);
      if ( !v9 )
        goto LABEL_61;
      v3 = v9;
    }
    while ( 1 )
    {
      sub_2DA06E0(*(_QWORD *)(v4 + 24));
      v5 = v4;
      v41 = v4;
      v4 = *(_QWORD *)(v4 + 16);
      v6 = *(volatile signed __int32 **)(v5 + 48);
      if ( v6 )
      {
        if ( &_pthread_key_create )
        {
          if ( _InterlockedExchangeAdd(v6 + 2, 0xFFFFFFFF) == 1 )
          {
LABEL_27:
            (*(void (**)(void))(*(_QWORD *)v6 + 16LL))();
            if ( &_pthread_key_create )
            {
              v14 = _InterlockedExchangeAdd(v6 + 3, 0xFFFFFFFF);
            }
            else
            {
              v14 = *((_DWORD *)v6 + 3);
              *((_DWORD *)v6 + 3) = v14 - 1;
            }
            if ( v14 == 1 )
              (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v6 + 24LL))(v6);
          }
        }
        else
        {
          v7 = *((_DWORD *)v6 + 2);
          *((_DWORD *)v6 + 2) = v7 - 1;
          if ( v7 == 1 )
            goto LABEL_27;
        }
      }
      j_j___libc_free_0(v41);
      if ( !v4 )
        goto LABEL_14;
    }
  }
}
