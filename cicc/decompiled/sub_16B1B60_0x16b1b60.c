// Function: sub_16B1B60
// Address: 0x16b1b60
//
__int64 __fastcall sub_16B1B60(
        __int64 a1,
        void (__fastcall *a2)(_BYTE *, size_t, __int64, __int64, _QWORD),
        __int64 *a3,
        unsigned __int8 a4,
        char a5)
{
  unsigned int v6; // r15d
  unsigned int v7; // ecx
  __int64 v8; // rax
  unsigned int v9; // r13d
  size_t v10; // rax
  __int64 v11; // rbx
  __int64 v12; // rdi
  __int64 v14; // rdi
  __int64 v15; // rsi
  __int64 v16; // rdx
  __int64 v17; // r9
  char *v18; // r12
  __int64 v19; // rsi
  unsigned __int64 v20; // rdx
  signed __int64 v21; // r9
  __int64 v22; // rax
  unsigned __int64 v23; // r10
  __int64 v24; // rdi
  size_t v25; // rdx
  char *v26; // r8
  char *v27; // r11
  __int64 v28; // rcx
  unsigned int v29; // esi
  __int64 i; // rax
  __int64 v31; // rcx
  size_t v32; // rax
  char *v33; // r10
  void *v34; // rdi
  size_t v35; // [rsp+0h] [rbp-A0h]
  char *v36; // [rsp+8h] [rbp-98h]
  char *v37; // [rsp+8h] [rbp-98h]
  char *v38; // [rsp+10h] [rbp-90h]
  char *v39; // [rsp+10h] [rbp-90h]
  __int64 v40; // [rsp+18h] [rbp-88h]
  signed __int64 v41; // [rsp+18h] [rbp-88h]
  char *v42; // [rsp+18h] [rbp-88h]
  char *v43; // [rsp+20h] [rbp-80h]
  char *v44; // [rsp+20h] [rbp-80h]
  signed __int64 v45; // [rsp+20h] [rbp-80h]
  signed __int64 v46; // [rsp+20h] [rbp-80h]
  size_t v47; // [rsp+28h] [rbp-78h]
  __int64 v48; // [rsp+28h] [rbp-78h]
  __int64 v49; // [rsp+28h] [rbp-78h]
  signed __int64 v50; // [rsp+28h] [rbp-78h]
  char *v51; // [rsp+30h] [rbp-70h]
  signed __int64 v52; // [rsp+30h] [rbp-70h]
  __int64 v53; // [rsp+30h] [rbp-70h]
  char *v54; // [rsp+30h] [rbp-70h]
  __int64 v55; // [rsp+30h] [rbp-70h]
  char *v56; // [rsp+38h] [rbp-68h]
  int v57; // [rsp+38h] [rbp-68h]
  unsigned __int8 v58; // [rsp+4Dh] [rbp-53h]
  void *src; // [rsp+60h] [rbp-40h] BYREF
  __int64 v64; // [rsp+68h] [rbp-38h]
  _BYTE v65[48]; // [rsp+70h] [rbp-30h] BYREF

  if ( *((_DWORD *)a3 + 2) )
  {
    v58 = 1;
    v6 = 0;
    v7 = 0;
    v8 = 0;
    while ( 1 )
    {
      while ( 1 )
      {
        v11 = 8 * v8;
        v12 = *(_QWORD *)(*a3 + 8 * v8);
        if ( v12 )
        {
          if ( *(_BYTE *)v12 == 64 )
            break;
        }
        v8 = ++v6;
        if ( v6 == *((_DWORD *)a3 + 2) )
          return v58;
      }
      v9 = v7 + 1;
      if ( v7 > 0x14 )
        break;
      v64 = 0;
      src = v65;
      v10 = strlen((const char *)(v12 + 1));
      if ( (unsigned __int8)sub_16B1570(v12 + 1, v10, a1, a2, (__int64)&src, a4, a5) )
      {
        v14 = *a3 + v11;
        v15 = *((unsigned int *)a3 + 2);
        v16 = *a3 + 8 * v15;
        if ( v16 != v14 + 8 )
        {
          memmove((void *)v14, (const void *)(v14 + 8), v16 - (v14 + 8));
          LODWORD(v15) = *((_DWORD *)a3 + 2);
        }
        v17 = (unsigned int)v64;
        v18 = (char *)src;
        v19 = (unsigned int)(v15 - 1);
        *((_DWORD *)a3 + 2) = v19;
        v20 = *((unsigned int *)a3 + 3);
        v21 = 8 * v17;
        v56 = &v18[v21];
        v22 = 8 * v19;
        v23 = v21 >> 3;
        if ( v11 == 8 * v19 )
        {
          if ( v20 - v19 < v23 )
          {
            v50 = v21;
            v55 = v21 >> 3;
            sub_16CD150(a3, a3 + 2, v19 + v23, 8);
            v19 = *((unsigned int *)a3 + 2);
            v21 = v50;
            LODWORD(v23) = v55;
            v22 = 8 * v19;
          }
          if ( v18 != v56 )
          {
            v57 = v23;
            memcpy((void *)(*a3 + v22), v18, v21);
            LODWORD(v19) = *((_DWORD *)a3 + 2);
            LODWORD(v23) = v57;
          }
          *((_DWORD *)a3 + 2) = v23 + v19;
        }
        else
        {
          if ( v19 + v23 > v20 )
          {
            v48 = v21 >> 3;
            v52 = v21;
            sub_16CD150(a3, a3 + 2, v19 + v23, 8);
            v19 = *((unsigned int *)a3 + 2);
            LODWORD(v23) = v48;
            v21 = v52;
            v22 = 8 * v19;
          }
          v24 = *a3;
          v25 = v22 - v11;
          v26 = (char *)(*a3 + v11);
          v27 = (char *)(*a3 + v22);
          v28 = (v22 - v11) >> 3;
          if ( v21 <= (unsigned __int64)(v22 - v11) )
          {
            v31 = v22 - v21;
            v32 = v21;
            v33 = (char *)(v24 + v31);
            v49 = v31;
            v34 = v27;
            v53 = v21 >> 3;
            if ( v21 >> 3 > (unsigned __int64)*((unsigned int *)a3 + 3) - v19 )
            {
              v35 = v21;
              v37 = v33;
              v39 = v27;
              v42 = (char *)(*a3 + v11);
              v46 = v21;
              sub_16CD150(a3, a3 + 2, v19 + (v21 >> 3), 8);
              v19 = *((unsigned int *)a3 + 2);
              v32 = v35;
              v33 = v37;
              v27 = v39;
              v26 = v42;
              v34 = (void *)(*a3 + 8 * v19);
              v21 = v46;
            }
            if ( v27 != v33 )
            {
              v36 = v27;
              v38 = v26;
              v41 = v21;
              v44 = v33;
              memmove(v34, v33, v32);
              LODWORD(v19) = *((_DWORD *)a3 + 2);
              v33 = v44;
              v27 = v36;
              v26 = v38;
              v21 = v41;
            }
            *((_DWORD *)a3 + 2) = v53 + v19;
            if ( v26 != v33 )
            {
              v45 = v21;
              v54 = v26;
              memmove(&v27[-(v49 - v11)], v26, v49 - v11);
              v21 = v45;
              v26 = v54;
            }
            if ( v18 != v56 )
              memmove(v26, v18, v21);
          }
          else
          {
            v29 = v23 + v19;
            *((_DWORD *)a3 + 2) = v29;
            if ( v26 != v27 )
            {
              v40 = (v22 - v11) >> 3;
              v43 = v27;
              v47 = v22 - v11;
              v51 = v26;
              memcpy((void *)(v24 + 8LL * v29 - v25), v26, v25);
              v28 = v40;
              v27 = v43;
              v25 = v47;
              v26 = v51;
            }
            if ( v28 )
            {
              for ( i = 0; i != v28; ++i )
                *(_QWORD *)&v26[8 * i] = *(_QWORD *)&v18[8 * i];
              v18 += v25;
            }
            if ( v56 != v18 )
              memcpy(v27, v18, v56 - v18);
          }
        }
        if ( src != v65 )
          _libc_free((unsigned __int64)src);
        v7 = v9;
      }
      else
      {
        ++v6;
        if ( src != v65 )
          _libc_free((unsigned __int64)src);
        v58 = 0;
        v7 = v9;
      }
      v8 = v6;
      if ( v6 == *((_DWORD *)a3 + 2) )
        return v58;
    }
    return 0;
  }
  else
  {
    return 1;
  }
}
