// Function: sub_2B7C3E0
// Address: 0x2b7c3e0
//
__int64 ***__fastcall sub_2B7C3E0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int64 a5)
{
  __int64 v5; // r9
  __int64 v8; // rax
  int v9; // ecx
  __int64 v10; // rsi
  __int64 v11; // rax
  int v12; // edx
  __int64 v13; // rdx
  __int64 ***v14; // r12
  __int64 v16; // rax
  unsigned int v17; // edx
  __int64 v18; // rsi
  __int64 v19; // rdx
  unsigned int v20; // eax
  unsigned int v21; // eax
  unsigned int v22; // edx
  __int64 v23; // rcx
  unsigned __int64 v24; // r8
  __int64 v25; // r9
  __int64 v26; // r10
  unsigned __int64 v27; // [rsp+8h] [rbp-148h]
  __int64 v28; // [rsp+8h] [rbp-148h]
  size_t n; // [rsp+10h] [rbp-140h]
  unsigned __int64 v30; // [rsp+18h] [rbp-138h]
  size_t v31; // [rsp+18h] [rbp-138h]
  int v32; // [rsp+18h] [rbp-138h]
  void *s; // [rsp+20h] [rbp-130h] BYREF
  __int64 v34; // [rsp+28h] [rbp-128h]
  _DWORD v35[12]; // [rsp+30h] [rbp-120h] BYREF
  void *v36; // [rsp+60h] [rbp-F0h] BYREF
  __int64 v37; // [rsp+68h] [rbp-E8h]
  _DWORD v38[12]; // [rsp+70h] [rbp-E0h] BYREF
  __int64 v39; // [rsp+A0h] [rbp-B0h] BYREF
  char v40; // [rsp+A8h] [rbp-A8h]
  _BYTE *v41; // [rsp+B0h] [rbp-A0h]
  __int64 v42; // [rsp+B8h] [rbp-98h]
  _BYTE v43[48]; // [rsp+C0h] [rbp-90h] BYREF
  _BYTE *v44; // [rsp+F0h] [rbp-60h]
  __int64 v45; // [rsp+F8h] [rbp-58h]
  _BYTE v46[16]; // [rsp+100h] [rbp-50h] BYREF
  __int64 v47; // [rsp+110h] [rbp-40h]
  __int64 v48; // [rsp+118h] [rbp-38h]

  v5 = a2;
  s = v35;
  v34 = 0xC00000000LL;
  if ( a5 > 0xC )
  {
    v30 = a5;
    sub_C8D5F0((__int64)&s, v35, a5, 4u, a5, a2);
    v24 = v30;
    v25 = a2;
    v26 = 4 * v30;
    if ( 4 * v30 )
    {
      v27 = v30;
      v31 = 4 * v30;
      memset(s, 255, v31);
      v26 = v31;
      v25 = a2;
      v24 = v27;
    }
    v28 = v25;
    n = v26;
    LODWORD(v34) = v24;
    v32 = v24;
    v36 = v38;
    v37 = 0xC00000000LL;
    sub_C8D5F0((__int64)&v36, v38, v24, 4u, v24, v25);
    LODWORD(a5) = v32;
    v5 = v28;
    if ( n )
    {
      memset(v36, 255, n);
      v5 = v28;
      LODWORD(a5) = v32;
    }
  }
  else if ( a5 )
  {
    v16 = 4 * a5;
    if ( 4 * a5 )
    {
      if ( (unsigned int)v16 < 8 )
      {
        if ( (v16 & 4) != 0 )
        {
          v35[0] = -1;
          *(_DWORD *)((char *)&v35[-1] + (unsigned int)v16) = -1;
        }
        else if ( (_DWORD)v16 )
        {
          LOBYTE(v35[0]) = -1;
        }
      }
      else
      {
        *(_QWORD *)((char *)&v35[-2] + (unsigned int)v16) = -1;
        if ( (unsigned int)(v16 - 1) >= 8 )
        {
          v17 = 0;
          do
          {
            v18 = v17;
            v17 += 8;
            *(_QWORD *)((char *)v35 + v18) = -1;
          }
          while ( v17 < (((_DWORD)v16 - 1) & 0xFFFFFFF8) );
        }
      }
    }
    LODWORD(v34) = a5;
    v36 = v38;
    HIDWORD(v37) = 12;
    if ( v16 )
    {
      if ( (unsigned int)v16 < 8 )
      {
        if ( (v16 & 4) != 0 )
        {
          v38[0] = -1;
          *(_DWORD *)((char *)&v38[-1] + (unsigned int)v16) = -1;
        }
        else if ( (_DWORD)v16 )
        {
          LOBYTE(v38[0]) = -1;
        }
      }
      else
      {
        v19 = (unsigned int)v16;
        v20 = v16 - 1;
        *(_QWORD *)((char *)&v38[-2] + v19) = -1;
        if ( v20 >= 8 )
        {
          v21 = v20 & 0xFFFFFFF8;
          v22 = 0;
          do
          {
            v23 = v22;
            v22 += 8;
            *(_QWORD *)((char *)v38 + v23) = -1;
          }
          while ( v22 < v21 );
        }
      }
    }
  }
  else
  {
    LODWORD(v34) = 0;
    v36 = v38;
    HIDWORD(v37) = 12;
  }
  v8 = *(_QWORD *)(v5 + 8);
  LODWORD(v37) = a5;
  v9 = *(_DWORD *)(v8 + 32);
  if ( (int)a5 > 0 )
  {
    v10 = 4LL * (unsigned int)(a5 - 1) + 4;
    v11 = 0;
    do
    {
      while ( 1 )
      {
        v12 = *(_DWORD *)(a4 + v11);
        if ( v12 >= v9 )
          break;
        *(_DWORD *)((char *)s + v11) = v12;
        v11 += 4;
        if ( v11 == v10 )
          goto LABEL_9;
      }
      *(_DWORD *)((char *)v36 + v11) = v12 - v9;
      v11 += 4;
    }
    while ( v11 != v10 );
LABEL_9:
    v8 = *(_QWORD *)(v5 + 8);
  }
  v13 = *a1;
  v39 = *(_QWORD *)(v8 + 24);
  v42 = 0xC00000000LL;
  v45 = 0x200000000LL;
  v48 = v13;
  v40 = 0;
  v41 = v43;
  v44 = v46;
  v47 = v13 + 3368;
  sub_2B7BF50((__int64)&v39, v5, (char *)s, (unsigned int)v34);
  if ( a3 )
    sub_2B7BF50((__int64)&v39, a3, (char *)v36, (unsigned int)v37);
  v14 = sub_2B7B8F0((__int64)&v39, 0, 0, 0, 0, 0, 0, 0, 0, 0);
  if ( v44 != v46 )
    _libc_free((unsigned __int64)v44);
  if ( v41 != v43 )
    _libc_free((unsigned __int64)v41);
  if ( v36 != v38 )
    _libc_free((unsigned __int64)v36);
  if ( s != v35 )
    _libc_free((unsigned __int64)s);
  return v14;
}
