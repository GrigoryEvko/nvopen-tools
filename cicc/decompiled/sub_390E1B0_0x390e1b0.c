// Function: sub_390E1B0
// Address: 0x390e1b0
//
double __fastcall sub_390E1B0(__int64 a1, __int64 **a2, __int64 a3, _QWORD *a4)
{
  double result; // xmm0_8
  _QWORD *v7; // rax
  unsigned __int64 v8; // rbx
  int v9; // r8d
  int v10; // r9d
  __int64 v11; // rax
  __int64 *v12; // rax
  __int64 *v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // rdx
  __int64 v16; // rcx
  int v17; // r9d
  double v18; // xmm0_8
  unsigned __int64 v19; // rdx
  __int64 v20; // rax
  __int64 *v21; // r9
  unsigned __int64 v22; // r12
  size_t v23; // r11
  int v24; // r12d
  double (__fastcall **v25)(__int64, _BYTE **, __int64, _QWORD *); // rax
  __int64 *v26; // [rsp-F8h] [rbp-F8h]
  __int64 v27; // [rsp-E0h] [rbp-E0h]
  __int64 *v28; // [rsp-D8h] [rbp-D8h] BYREF
  __int64 v29; // [rsp-D0h] [rbp-D0h]
  _BYTE v30[64]; // [rsp-C8h] [rbp-C8h] BYREF
  _BYTE *v31; // [rsp-88h] [rbp-88h] BYREF
  __int64 v32; // [rsp-80h] [rbp-80h]
  _BYTE v33[120]; // [rsp-78h] [rbp-78h] BYREF

  result = 0.0;
  if ( *((_DWORD *)a2 + 2) )
  {
    v27 = sub_390E170(a1, **a2, a3, a4);
    v28 = (__int64 *)v30;
    v29 = 0x800000000LL;
    v7 = (_QWORD *)**a2;
    if ( v7 != *(_QWORD **)(v7[3] + 104LL) )
    {
      v8 = *v7 & 0xFFFFFFFFFFFFFFF8LL;
      if ( v8 )
      {
        do
        {
          if ( *(_BYTE *)(v8 + 16) == 9 && (*(_QWORD *)(v8 + 48) & *(_QWORD *)(a1 + 8)) != 0 )
          {
            if ( v27 != sub_390E170(a1, v8, a3, a4) )
              break;
            v11 = (unsigned int)v29;
            if ( (unsigned int)v29 >= HIDWORD(v29) )
            {
              sub_16CD150((__int64)&v28, v30, 0, 8, v9, v10);
              v11 = (unsigned int)v29;
            }
            v28[v11] = v8;
            LODWORD(v29) = v29 + 1;
          }
          if ( v8 == *(_QWORD *)(*(_QWORD *)(v8 + 24) + 104LL) )
            break;
          v8 = *(_QWORD *)v8 & 0xFFFFFFFFFFFFFFF8LL;
        }
        while ( v8 );
        v12 = v28;
        v13 = &v28[(unsigned int)v29];
        if ( v28 != v13 )
        {
          while ( v12 < --v13 )
          {
            v14 = *v12++;
            *(v12 - 1) = *v13;
            *v13 = v14;
          }
        }
      }
    }
    v18 = (**(double (__fastcall ***)(__int64, __int64 **, __int64, _QWORD *, double))a1)(a1, &v28, a3, a4, 0.0);
    v31 = v33;
    v32 = 0x800000000LL;
    if ( (_DWORD)v29 )
    {
      sub_390DB40((__int64)&v31, (__int64)&v28, v15, v16, (int)&v31, v17);
      v20 = (unsigned int)v32;
      v19 = HIDWORD(v32) - (unsigned __int64)(unsigned int)v32;
    }
    else
    {
      v19 = 8;
      v20 = 0;
    }
    v21 = *a2;
    v22 = *((unsigned int *)a2 + 2);
    v23 = 8 * v22;
    if ( v22 > v19 )
    {
      v26 = *a2;
      sub_16CD150((__int64)&v31, v33, v22 + v20, 8, (int)&v31, (int)v21);
      v20 = (unsigned int)v32;
      v23 = 8 * v22;
      v21 = v26;
    }
    if ( v23 )
    {
      memcpy(&v31[8 * v20], v21, v23);
      LODWORD(v20) = v32;
    }
    v24 = v20 + v22;
    v25 = *(double (__fastcall ***)(__int64, _BYTE **, __int64, _QWORD *))a1;
    LODWORD(v32) = v24;
    result = (*v25)(a1, &v31, a3, a4) - v18;
    if ( v31 != v33 )
      _libc_free((unsigned __int64)v31);
    if ( v28 != (__int64 *)v30 )
      _libc_free((unsigned __int64)v28);
  }
  return result;
}
