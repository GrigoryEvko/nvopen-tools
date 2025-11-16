// Function: sub_2F8ADD0
// Address: 0x2f8add0
//
void __fastcall sub_2F8ADD0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r12
  unsigned int v8; // eax
  __int64 v9; // rcx
  __int64 v10; // rbx
  __int64 v11; // r14
  __int64 v12; // rdx
  __int64 v13; // r14
  __int64 v14; // rdx
  __int64 v15; // rbx
  __int64 v16; // rcx
  __int64 v17; // r15
  __int64 v18; // rcx
  char *v19; // rdi
  __int64 v20; // rcx
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // rdi
  char *v24; // rdi
  __int64 v25; // [rsp+8h] [rbp-B8h]
  __int64 v26; // [rsp+20h] [rbp-A0h]
  __int64 v27; // [rsp+30h] [rbp-90h]
  unsigned int v28; // [rsp+38h] [rbp-88h]
  char v29; // [rsp+3Ch] [rbp-84h]
  char v30; // [rsp+3Ch] [rbp-84h]
  char *v31; // [rsp+40h] [rbp-80h] BYREF
  __int64 v32; // [rsp+48h] [rbp-78h]
  _BYTE v33[48]; // [rsp+50h] [rbp-70h] BYREF
  int v34; // [rsp+80h] [rbp-40h]

  if ( a1 != a2 && a2 != a1 + 88 )
  {
    v7 = a1 + 104;
    v25 = a1 + 16;
    do
    {
      v8 = *(_DWORD *)(v7 - 8);
      v9 = *(_QWORD *)(v7 - 16);
      v10 = v7 - 16;
      v11 = v7;
      v12 = *(unsigned int *)(v7 + 8);
      v27 = v9;
      v28 = v8;
      if ( v8 <= *(_DWORD *)(a1 + 8) )
      {
        v20 = *(unsigned __int8 *)(v7 - 4);
        v31 = v33;
        v30 = v20;
        v32 = 0x600000000LL;
        if ( (_DWORD)v12 )
        {
          sub_2F8ABB0((__int64)&v31, (char **)v7, v12, v20, a5, a6);
          v8 = v28;
        }
        v21 = *(unsigned int *)(v7 + 64);
        v34 = *(_DWORD *)(v7 + 64);
        if ( v8 > *(_DWORD *)(v7 - 96) )
        {
          do
          {
            v22 = *(_QWORD *)(v11 - 104);
            v23 = v11;
            v10 = v11 - 104;
            v11 -= 88;
            *(_QWORD *)(v11 + 72) = v22;
            *(_DWORD *)(v11 + 80) = *(_DWORD *)(v11 - 8);
            *(_BYTE *)(v11 + 84) = *(_BYTE *)(v11 - 4);
            sub_2F8ABB0(v23, (char **)v11, v21, v20, a5, a6);
            *(_DWORD *)(v11 + 152) = *(_DWORD *)(v11 + 64);
            v8 = v28;
          }
          while ( v28 > *(_DWORD *)(v11 - 96) );
        }
        *(_DWORD *)(v10 + 8) = v8;
        *(_QWORD *)v10 = v27;
        *(_BYTE *)(v10 + 12) = v30;
        sub_2F8ABB0(v11, &v31, v27, v20, a5, a6);
        v24 = v31;
        *(_DWORD *)(v10 + 80) = v34;
        if ( v24 != v33 )
          _libc_free((unsigned __int64)v24);
        v26 = v7 + 72;
      }
      else
      {
        v29 = *(_BYTE *)(v7 - 4);
        v31 = v33;
        v32 = 0x600000000LL;
        if ( (_DWORD)v12 )
          sub_2F8ABB0((__int64)&v31, (char **)v7, v12, v9, a5, a6);
        v13 = v7;
        v14 = v10 - a1;
        v34 = *(_DWORD *)(v7 + 64);
        v26 = v7 + 72;
        v15 = 0x2E8BA2E8BA2E8BA3LL * ((v10 - a1) >> 3);
        if ( v14 > 0 )
        {
          do
          {
            v16 = *(_QWORD *)(v13 - 104);
            v17 = v13;
            v13 -= 88;
            *(_QWORD *)(v13 + 72) = v16;
            *(_DWORD *)(v13 + 80) = *(_DWORD *)(v13 - 8);
            v18 = *(unsigned __int8 *)(v13 - 4);
            *(_BYTE *)(v13 + 84) = v18;
            sub_2F8ABB0(v17, (char **)v13, v14, v18, a5, a6);
            v9 = *(unsigned int *)(v17 - 24);
            *(_DWORD *)(v17 + 64) = v9;
            --v15;
          }
          while ( v15 );
        }
        *(_QWORD *)a1 = v27;
        *(_DWORD *)(a1 + 8) = v28;
        *(_BYTE *)(a1 + 12) = v29;
        sub_2F8ABB0(v25, &v31, v14, v9, a5, a6);
        v19 = v31;
        *(_DWORD *)(a1 + 80) = v34;
        if ( v19 != v33 )
          _libc_free((unsigned __int64)v19);
      }
      v7 += 88;
    }
    while ( a2 != v26 );
  }
}
