// Function: sub_2EB6B70
// Address: 0x2eb6b70
//
__int64 __fastcall sub_2EB6B70(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rax
  __int64 *v6; // r15
  __int64 v7; // r14
  __int64 *v8; // r13
  __int64 *i; // r15
  unsigned int v10; // eax
  void *v11; // rax
  _QWORD *v12; // rdi
  _BYTE *v13; // rax
  __int64 *v14; // rdi
  __int64 result; // rax
  unsigned int v16; // r9d
  const void *v17; // r10
  char *v18; // rdi
  size_t v19; // r11
  __int64 v20; // r9
  char *v21; // rcx
  unsigned __int64 v22; // rsi
  __int64 v23; // rsi
  __int64 v24; // rax
  __int64 v25; // rsi
  void *v26; // rax
  _QWORD *v27; // rdi
  _BYTE *v28; // rax
  __int64 *v29; // rdi
  char *j; // r9
  __int64 v31; // rdx
  __int64 v32; // rsi
  char *v33; // rax
  char *v34; // rdi
  unsigned __int64 v35; // [rsp-C8h] [rbp-C8h]
  __int64 v36; // [rsp-C8h] [rbp-C8h]
  char *v37; // [rsp-C0h] [rbp-C0h]
  const void *v38; // [rsp-C0h] [rbp-C0h]
  unsigned int v39; // [rsp-B8h] [rbp-B8h]
  char *v40; // [rsp-B8h] [rbp-B8h]
  unsigned int v41; // [rsp-B8h] [rbp-B8h]
  __int64 v42[4]; // [rsp-A8h] [rbp-A8h] BYREF
  char *v43; // [rsp-88h] [rbp-88h] BYREF
  __int64 v44; // [rsp-80h] [rbp-80h]
  _BYTE v45[120]; // [rsp-78h] [rbp-78h] BYREF

  if ( !*(_BYTE *)(a1 + 136) || !*(_QWORD *)(a1 + 128) )
    return 1;
  v5 = *(unsigned int *)(a1 + 56);
  if ( !(_DWORD)v5 )
    BUG();
  v6 = *(__int64 **)(a1 + 48);
  v7 = *v6;
  if ( *(_DWORD *)(*v6 + 72) )
  {
    v26 = sub_CB72A0();
    sub_904010((__int64)v26, "DFSIn number for the tree root is not:\n\t");
    sub_2EB3840(v7);
    v27 = sub_CB72A0();
    v28 = (_BYTE *)v27[4];
    if ( (unsigned __int64)v28 >= v27[3] )
    {
      sub_CB5D20((__int64)v27, 10);
    }
    else
    {
      v27[4] = v28 + 1;
      *v28 = 10;
    }
    v29 = (__int64 *)sub_CB72A0();
    result = 0;
    if ( v29[2] != v29[4] )
    {
      sub_CB5AE0(v29);
      return 0;
    }
  }
  else
  {
    v8 = &v6[v5];
    for ( i = v6 + 1; ; ++i )
    {
      if ( v7 )
      {
        v10 = *(_DWORD *)(v7 + 32);
        if ( v10 )
        {
          v16 = *(_DWORD *)(v7 + 32);
          v44 = 0x800000000LL;
          v17 = *(const void **)(v7 + 24);
          v18 = v45;
          v19 = 8LL * v10;
          v43 = v45;
          if ( v10 > 8uLL )
          {
            v36 = 8LL * v10;
            v38 = v17;
            v41 = v10;
            sub_C8D5F0((__int64)&v43, v45, v10, 8u, a5, v10);
            v16 = v41;
            v17 = v38;
            v19 = v36;
            v18 = &v43[8 * (unsigned int)v44];
          }
          v39 = v16;
          memcpy(v18, v17, v19);
          LODWORD(v44) = v44 + v39;
          v20 = 8LL * (unsigned int)v44;
          v21 = &v43[v20];
          if ( v43 != &v43[v20] )
          {
            v35 = 8LL * (unsigned int)v44;
            v37 = &v43[v20];
            _BitScanReverse64(&v22, v20 >> 3);
            v40 = v43;
            sub_2EB69B0(v43, (__int64 *)&v43[v20], 2LL * (int)(63 - (v22 ^ 0x3F)));
            if ( v35 > 0x80 )
            {
              sub_2EB2F50(v40, v40 + 128);
              for ( j = v40 + 128; v37 != j; *(_QWORD *)v34 = v32 )
              {
                v31 = *((_QWORD *)j - 1);
                v32 = *(_QWORD *)j;
                v33 = j - 8;
                if ( *(_DWORD *)(*(_QWORD *)j + 72LL) >= *(_DWORD *)(v31 + 72) )
                {
                  v34 = j;
                }
                else
                {
                  do
                  {
                    *((_QWORD *)v33 + 1) = v31;
                    v34 = v33;
                    v31 = *((_QWORD *)v33 - 1);
                    v33 -= 8;
                    a5 = *(unsigned int *)(v31 + 72);
                  }
                  while ( *(_DWORD *)(v32 + 72) < (unsigned int)a5 );
                }
                j += 8;
              }
            }
            else
            {
              sub_2EB2F50(v40, v37);
            }
            v21 = v43;
          }
          v42[0] = v7;
          v42[2] = 0;
          v42[1] = (__int64)&v43;
          v23 = *(_QWORD *)v21;
          if ( *(_DWORD *)(v7 + 72) + 1 != *(_DWORD *)(*(_QWORD *)v21 + 72LL)
            || (v23 = *(_QWORD *)&v21[8 * (unsigned int)v44 - 8], *(_DWORD *)(v23 + 76) + 1 != *(_DWORD *)(v7 + 76)) )
          {
            sub_2EB45B0(v42, v23, 0);
LABEL_37:
            if ( v43 != v45 )
              _libc_free((unsigned __int64)v43);
            return 0;
          }
          v24 = 0;
          while ( v24 != (unsigned int)v44 - 1LL )
          {
            v25 = *(_QWORD *)&v21[8 * v24++];
            if ( *(_DWORD *)(v25 + 76) + 1 != *(_DWORD *)(*(_QWORD *)&v21[8 * v24] + 72LL) )
            {
              sub_2EB45B0(v42, v25, *(_QWORD *)&v21[8 * v24]);
              goto LABEL_37;
            }
          }
          if ( v21 != v45 )
            _libc_free((unsigned __int64)v21);
        }
        else if ( *(_DWORD *)(v7 + 72) + 1 != *(_DWORD *)(v7 + 76) )
        {
          v11 = sub_CB72A0();
          sub_904010((__int64)v11, "Tree leaf should have DFSOut = DFSIn + 1:\n\t");
          sub_2EB3840(v7);
          v12 = sub_CB72A0();
          v13 = (_BYTE *)v12[4];
          if ( (unsigned __int64)v13 >= v12[3] )
          {
            sub_CB5D20((__int64)v12, 10);
          }
          else
          {
            v12[4] = v13 + 1;
            *v13 = 10;
          }
          v14 = (__int64 *)sub_CB72A0();
          if ( v14[4] != v14[2] )
            sub_CB5AE0(v14);
          return 0;
        }
      }
      if ( v8 == i )
        break;
      v7 = *i;
    }
    return 1;
  }
  return result;
}
