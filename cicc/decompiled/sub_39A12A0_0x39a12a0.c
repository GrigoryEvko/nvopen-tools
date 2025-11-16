// Function: sub_39A12A0
// Address: 0x39a12a0
//
void __fastcall sub_39A12A0(__int64 **a1, __int64 a2, __int64 a3, __int64 a4, char a5)
{
  __int64 *v7; // rcx
  int v8; // r9d
  unsigned __int64 v9; // r12
  __int64 **v10; // rdi
  __int64 ***v11; // r8
  int v12; // esi
  __int64 **v13; // rax
  __int64 **v14; // r13
  __int64 **v15; // r14
  __int64 v16; // rdi
  void (*v17)(); // rcx
  int v18; // edx
  _QWORD *v19; // rsi
  void (*v20)(); // rcx
  __int64 v21; // rsi
  __int64 v22; // rdx
  __int64 **v23; // r13
  __int64 *v24; // r12
  __int64 v25; // rax
  __int64 *v26; // rdi
  __int64 v27; // rax
  __int64 *v28; // rax
  __int64 v29; // rdx
  __int64 *v30; // rdi
  __int64 v31; // rax
  __int64 *v32; // rax
  __int64 v33; // rdx
  __int64 **v36; // [rsp+28h] [rbp-288h]
  _QWORD v37[4]; // [rsp+30h] [rbp-280h] BYREF
  _QWORD v38[2]; // [rsp+50h] [rbp-260h] BYREF
  __int16 v39; // [rsp+60h] [rbp-250h]
  __int64 **v40; // [rsp+70h] [rbp-240h] BYREF
  __int64 v41; // [rsp+78h] [rbp-238h]
  _BYTE s[560]; // [rsp+80h] [rbp-230h] BYREF

  if ( *((_DWORD *)a1 + 3) )
  {
    (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(a2 + 256) + 160LL))(*(_QWORD *)(a2 + 256), a3, 0);
    v9 = *((unsigned int *)a1 + 3);
    v10 = (__int64 **)s;
    v11 = &v40;
    v40 = (__int64 **)s;
    v41 = 0x4000000000LL;
    if ( (unsigned int)v9 > 0x40 )
    {
      sub_16CD150((__int64)&v40, s, v9, 8, (int)&v40, v8);
      v10 = v40;
    }
    LODWORD(v41) = v9;
    if ( 8 * v9 )
      memset(v10, 0, 8 * v9);
    v12 = *((_DWORD *)a1 + 2);
    if ( v12 )
    {
      v26 = *a1;
      v27 = **a1;
      if ( v27 != -8 && v27 )
      {
        v7 = *a1;
      }
      else
      {
        v28 = v26 + 1;
        do
        {
          do
          {
            v29 = *v28;
            v7 = v28++;
          }
          while ( v29 == -8 );
        }
        while ( !v29 );
      }
      v30 = &v26[v12];
      while ( v7 != v30 )
      {
        v40[*(unsigned int *)(*v7 + 20)] = (__int64 *)*v7;
        v31 = v7[1];
        if ( v31 && v31 != -8 )
        {
          ++v7;
        }
        else
        {
          v32 = v7 + 2;
          do
          {
            do
            {
              v33 = *v32;
              v7 = v32++;
            }
            while ( v33 == -8 );
          }
          while ( !v33 );
        }
      }
    }
    v36 = &v40[(unsigned int)v41];
    if ( v40 != v36 )
    {
      v13 = a1;
      v14 = v40;
      v15 = v13;
      do
      {
        v16 = *(_QWORD *)(a2 + 256);
        v20 = *(void (**)())(*(_QWORD *)v16 + 104LL);
        v21 = **v14;
        v22 = (__int64)(*v14 + 3);
        v38[0] = v37;
        v37[0] = v22;
        v37[1] = v21;
        v39 = 261;
        if ( v20 != nullsub_580 )
        {
          ((void (__fastcall *)(__int64, _QWORD *, __int64, void (*)(), __int64 ***))v20)(v16, v38, 1, v20, v11);
          v16 = *(_QWORD *)(a2 + 256);
        }
        if ( *((_BYTE *)v15 + 52) )
        {
          (*(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v16 + 176LL))(v16, (*v14)[1], 0);
          v16 = *(_QWORD *)(a2 + 256);
        }
        v17 = *(void (**)())(*(_QWORD *)v16 + 104LL);
        v18 = *((_DWORD *)*v14 + 4);
        v38[0] = "string offset=";
        v39 = 2307;
        LODWORD(v37[0]) = v18;
        v38[1] = v37[0];
        if ( v17 != nullsub_580 )
        {
          ((void (__fastcall *)(__int64, _QWORD *, __int64, void (*)(), __int64 ***))v17)(v16, v38, 1, v17, v11);
          v16 = *(_QWORD *)(a2 + 256);
        }
        v19 = *v14++;
        (*(void (__fastcall **)(__int64, _QWORD *, __int64))(*(_QWORD *)v16 + 400LL))(v16, v19 + 3, *v19 + 1LL);
      }
      while ( v36 != v14 );
    }
    if ( a4 )
    {
      (*(void (__fastcall **)(_QWORD, __int64, _QWORD, __int64 *, __int64 ***))(**(_QWORD **)(a2 + 256) + 160LL))(
        *(_QWORD *)(a2 + 256),
        a4,
        0,
        v7,
        v11);
      v24 = (__int64 *)v40;
      v23 = &v40[(unsigned int)v41];
      if ( v40 == v23 )
      {
LABEL_18:
        if ( v23 != (__int64 **)s )
          _libc_free((unsigned __int64)v23);
        return;
      }
      do
      {
        v25 = *v24;
        if ( a5 )
          sub_397C4E0(a2, *(_QWORD *)(v25 + 8), *(_QWORD *)(v25 + 16));
        else
          (*(void (__fastcall **)(_QWORD, _QWORD, __int64))(**(_QWORD **)(a2 + 256) + 424LL))(
            *(_QWORD *)(a2 + 256),
            *(unsigned int *)(v25 + 16),
            4);
        ++v24;
      }
      while ( v23 != (__int64 **)v24 );
    }
    v23 = v40;
    goto LABEL_18;
  }
}
