// Function: sub_28A89D0
// Address: 0x28a89d0
//
__int64 __fastcall sub_28A89D0(__int64 a1, _BYTE *a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rbx
  __int64 v7; // rax
  bool v8; // zf
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  _BYTE *v13; // r14
  unsigned __int64 v14; // r12
  unsigned __int64 v15; // rdi
  void **v17; // rdx
  int v18; // edi
  void **v19; // rax
  int v20; // eax
  __int64 *v21; // rax
  __int64 **v22; // rax
  __int64 **v23; // rdx
  __int64 v24; // [rsp+20h] [rbp-1B0h] BYREF
  __int64 **v25; // [rsp+28h] [rbp-1A8h]
  __int64 v26; // [rsp+30h] [rbp-1A0h]
  int v27; // [rsp+38h] [rbp-198h]
  char v28; // [rsp+3Ch] [rbp-194h]
  _BYTE v29[16]; // [rsp+40h] [rbp-190h] BYREF
  __int64 v30; // [rsp+50h] [rbp-180h] BYREF
  void **v31; // [rsp+58h] [rbp-178h]
  __int64 v32; // [rsp+60h] [rbp-170h]
  int v33; // [rsp+68h] [rbp-168h]
  char v34; // [rsp+6Ch] [rbp-164h]
  _BYTE v35[16]; // [rsp+70h] [rbp-160h] BYREF
  __int64 v36[9]; // [rsp+80h] [rbp-150h] BYREF
  __int64 v37; // [rsp+C8h] [rbp-108h]
  __int64 v38; // [rsp+D0h] [rbp-100h]
  unsigned int v39; // [rsp+D8h] [rbp-F8h]
  _BYTE *v40; // [rsp+E0h] [rbp-F0h]
  __int64 v41; // [rsp+E8h] [rbp-E8h]
  _BYTE v42[128]; // [rsp+F0h] [rbp-E0h] BYREF
  __int64 v43; // [rsp+170h] [rbp-60h]
  __int64 v44; // [rsp+178h] [rbp-58h]
  __int64 v45; // [rsp+180h] [rbp-50h]
  unsigned int v46; // [rsp+188h] [rbp-48h]
  _BYTE *v47; // [rsp+190h] [rbp-40h]
  __int64 v48; // [rsp+198h] [rbp-38h]
  _BYTE v49[48]; // [rsp+1A0h] [rbp-30h] BYREF

  v6 = a4;
  v7 = sub_BC1CD0(a4, &unk_4F89C30, a3);
  v8 = *a2 == 0;
  v36[0] = a3;
  if ( !v8 )
    v6 = 0;
  v36[2] = v7 + 8;
  v36[1] = sub_B2BEC0(a3);
  v36[3] = v6;
  v41 = 0x1000000000LL;
  memset(&v36[4], 0, 40);
  v37 = 0;
  v38 = 0;
  v39 = 0;
  v40 = v42;
  v43 = 0;
  v44 = 0;
  v45 = 0;
  v46 = 0;
  v47 = v49;
  v48 = 0;
  if ( !sub_28A58D0(v36, (__int64)&unk_4F89C30, v9, v10, v11, v12) )
  {
    *(_QWORD *)(a1 + 8) = a1 + 32;
    *(_QWORD *)a1 = 0;
    *(_QWORD *)(a1 + 16) = 2;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 56) = a1 + 80;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    sub_AE6EC0(a1, (__int64)&qword_4F82400);
    goto LABEL_9;
  }
  v8 = *a2 == 0;
  v24 = 0;
  v25 = (__int64 **)v29;
  v26 = 2;
  v27 = 0;
  v28 = 1;
  v30 = 0;
  v31 = (void **)v35;
  v32 = 2;
  v33 = 0;
  v34 = 1;
  if ( v8 )
  {
    sub_AE6EC0((__int64)&v24, (__int64)&unk_4F875F0);
    if ( v34 )
    {
      v17 = &v31[HIDWORD(v32)];
      v18 = HIDWORD(v32);
      if ( v31 == v17 )
      {
LABEL_30:
        v20 = v33;
      }
      else
      {
        v19 = v31;
        while ( *v19 != &unk_4F81450 )
        {
          if ( v17 == ++v19 )
            goto LABEL_30;
        }
        --HIDWORD(v32);
        *v19 = v31[HIDWORD(v32)];
        v18 = HIDWORD(v32);
        ++v30;
        v20 = v33;
      }
    }
    else
    {
      v21 = sub_C8CA60((__int64)&v30, (__int64)&unk_4F81450);
      if ( v21 )
      {
        *v21 = -2;
        ++v30;
        v18 = HIDWORD(v32);
        v20 = ++v33;
      }
      else
      {
        v18 = HIDWORD(v32);
        v20 = v33;
      }
    }
    if ( v20 == v18 )
    {
      if ( v28 )
      {
        v22 = v25;
        v23 = &v25[HIDWORD(v26)];
        if ( v25 != v23 )
        {
          while ( *v22 != &qword_4F82400 )
          {
            if ( v23 == ++v22 )
              goto LABEL_29;
          }
          goto LABEL_5;
        }
      }
      else if ( sub_C8CA60((__int64)&v24, (__int64)&qword_4F82400) )
      {
        goto LABEL_5;
      }
    }
LABEL_29:
    sub_AE6EC0((__int64)&v24, (__int64)&unk_4F81450);
  }
LABEL_5:
  sub_C8CF70(a1, (void *)(a1 + 32), 2, (__int64)v29, (__int64)&v24);
  sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v35, (__int64)&v30);
  if ( !v34 )
    _libc_free((unsigned __int64)v31);
  if ( !v28 )
    _libc_free((unsigned __int64)v25);
LABEL_9:
  v13 = v47;
  v14 = (unsigned __int64)&v47[176 * (unsigned int)v48];
  if ( v47 != (_BYTE *)v14 )
  {
    do
    {
      v14 -= 176LL;
      v15 = *(_QWORD *)(v14 + 8);
      if ( v15 != v14 + 24 )
        _libc_free(v15);
    }
    while ( v13 != (_BYTE *)v14 );
    v14 = (unsigned __int64)v47;
  }
  if ( (_BYTE *)v14 != v49 )
    _libc_free(v14);
  sub_C7D6A0(v44, 16LL * v46, 8);
  if ( v40 != v42 )
    _libc_free((unsigned __int64)v40);
  sub_C7D6A0(v37, 24LL * v39, 8);
  return a1;
}
