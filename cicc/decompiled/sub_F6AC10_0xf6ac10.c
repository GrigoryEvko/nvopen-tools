// Function: sub_F6AC10
// Address: 0xf6ac10
//
__int64 __fastcall sub_F6AC10(
        char *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 *a6,
        unsigned __int8 a7)
{
  unsigned __int64 v8; // rcx
  __int64 v10; // rdx
  __int64 v11; // r9
  _QWORD *v12; // rdi
  __int64 v13; // rax
  int v14; // ebx
  int v15; // ecx
  char *v16; // rsi
  char *v17; // rax
  char *v18; // r13
  size_t v19; // r15
  __int64 v20; // r12
  unsigned __int64 v21; // rdx
  unsigned int v22; // r15d
  __int64 v23; // rdi
  int v24; // eax
  unsigned int v25; // eax
  unsigned int v27; // [rsp-10h] [rbp-C0h]
  __int64 v30; // [rsp+18h] [rbp-98h]
  __int64 v31; // [rsp+20h] [rbp-90h]
  char *v34; // [rsp+48h] [rbp-68h]
  _QWORD *v35; // [rsp+50h] [rbp-60h] BYREF
  __int64 v36; // [rsp+58h] [rbp-58h]
  _QWORD v37[10]; // [rsp+60h] [rbp-50h] BYREF

  v8 = 4;
  v10 = 1;
  v11 = 0;
  v35 = v37;
  v37[0] = a1;
  v12 = v37;
  v13 = 0;
  v31 = a3;
  v14 = 0;
  v36 = 0x400000001LL;
  v30 = a4;
  while ( 1 )
  {
    v16 = (char *)v12[v13];
    v17 = (char *)*((_QWORD *)v16 + 2);
    v18 = (char *)*((_QWORD *)v16 + 1);
    v19 = v17 - v18;
    v20 = (v17 - v18) >> 3;
    v21 = v20 + v10;
    if ( v21 > v8 )
    {
      v16 = (char *)v37;
      v34 = v17;
      sub_C8D5F0((__int64)&v35, v37, v21, 8u, a5, v11);
      v12 = v35;
      v17 = v34;
    }
    v15 = v36;
    if ( v18 != v17 )
    {
      v16 = v18;
      memmove(&v12[(unsigned int)v36], v18, v19);
      v15 = v36;
      v12 = v35;
    }
    v13 = (unsigned int)(v14 + 1);
    LODWORD(v36) = v20 + v15;
    v10 = (unsigned int)(v20 + v15);
    v14 = v13;
    if ( (_DWORD)v13 == (_DWORD)v20 + v15 )
      break;
    v8 = HIDWORD(v36);
  }
  v22 = 0;
  if ( (_DWORD)v13 )
  {
    while ( 1 )
    {
      v23 = v12[(unsigned int)v10 - 1];
      LODWORD(v36) = v10 - 1;
      v16 = (char *)&v35;
      v24 = sub_F681E0(v23, (__int64)&v35, a2, v31, v30, a5, a6, a7);
      LODWORD(v10) = v36;
      v22 |= v24;
      v25 = v27;
      if ( !(_DWORD)v36 )
        break;
      v12 = v35;
    }
    if ( ((unsigned __int8)v22 & (v30 != 0)) != 0 )
    {
      v16 = a1;
      LOBYTE(v25) = v22 & (v30 != 0);
      v22 = v25;
      sub_DAC8B0(v30, a1);
    }
    v12 = v35;
  }
  if ( v12 != v37 )
    _libc_free(v12, v16);
  return v22;
}
