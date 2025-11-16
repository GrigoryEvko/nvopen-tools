// Function: sub_EDE4E0
// Address: 0xede4e0
//
__int64 *__fastcall sub_EDE4E0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v8; // rcx
  __int64 v9; // rax
  _WORD *v10; // rax
  int v11; // r8d
  _QWORD *v12; // rbx
  int v13; // ecx
  unsigned __int64 v14; // rax
  __int64 *v15; // rbx
  unsigned __int64 v16; // r8
  unsigned __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // r14
  __int64 v20; // r15
  __int64 v21; // rsi
  _BYTE *v22; // rdi
  _QWORD *v23; // rbx
  __int64 v24; // rax
  __int64 *v25; // rdx
  __int64 *v26; // rax
  __int64 *v27; // rdx
  _QWORD *v28; // r9
  __int64 v29; // r15
  __int64 *v30; // r8
  __int64 *v32; // [rsp+10h] [rbp-A0h]
  __int64 *v33; // [rsp+10h] [rbp-A0h]
  _QWORD *v34; // [rsp+10h] [rbp-A0h]
  __int64 *v35; // [rsp+10h] [rbp-A0h]
  unsigned __int64 v36; // [rsp+10h] [rbp-A0h]
  unsigned __int64 v37; // [rsp+10h] [rbp-A0h]
  _QWORD *v38; // [rsp+18h] [rbp-98h]
  __int64 v39; // [rsp+20h] [rbp-90h] BYREF
  __int64 *v40; // [rsp+28h] [rbp-88h]
  int v41; // [rsp+30h] [rbp-80h]
  int v42; // [rsp+34h] [rbp-7Ch]
  char v43; // [rsp+38h] [rbp-78h]
  _BYTE *v44; // [rsp+40h] [rbp-70h] BYREF
  __int64 v45; // [rsp+48h] [rbp-68h]
  _BYTE v46[96]; // [rsp+50h] [rbp-60h] BYREF

  *a1 = 0;
  a1[1] = 0;
  a1[2] = 0;
  v8 = *(_QWORD **)(a2 + 16);
  v9 = *(_QWORD *)(v8[2] + 8 * (a3 & (*v8 - 1LL)));
  if ( !v9 || (v10 = (_WORD *)(v8[3] + v9), v11 = (unsigned __int16)*v10, v12 = v10 + 1, !*v10) )
  {
LABEL_35:
    *(_QWORD *)a2 = a3;
    *(_BYTE *)(a2 + 8) = 1;
    return a1;
  }
  v13 = 0;
  while ( 1 )
  {
    v14 = v12[1];
    if ( a3 == *v12 && a3 == v12[2] )
      break;
    ++v13;
    v12 = (_QWORD *)((char *)v12 + v14 + 24);
    if ( v11 == v13 )
      goto LABEL_35;
  }
  v15 = v12 + 3;
  v16 = v14 >> 3;
  v44 = v46;
  v45 = 0x600000000LL;
  if ( v14 > 0x37 )
  {
    v37 = v14 >> 3;
    sub_C8D5F0((__int64)&v44, v46, v14 >> 3, 8u, v16, a6);
    v18 = (unsigned int)v45;
    v17 = HIDWORD(v45);
    v16 = v37;
LABEL_9:
    v19 = 0;
    while ( 1 )
    {
      v20 = *v15++;
      if ( v18 + 1 > v17 )
      {
        v36 = v16;
        sub_C8D5F0((__int64)&v44, v46, v18 + 1, 8u, v16, v18 + 1);
        v18 = (unsigned int)v45;
        v16 = v36;
      }
      ++v19;
      *(_QWORD *)&v44[8 * v18] = v20;
      v18 = (unsigned int)(v45 + 1);
      LODWORD(v45) = v45 + 1;
      if ( v16 == v19 )
        break;
      v17 = HIDWORD(v45);
    }
    v16 = (unsigned int)v18;
    goto LABEL_15;
  }
  if ( v16 )
  {
    v17 = 6;
    v18 = 0;
    goto LABEL_9;
  }
LABEL_15:
  v21 = v16;
  sub_ED87A0(a1, v16);
  v22 = v44;
  v38 = &v44[8 * (unsigned int)v45];
  if ( v38 != (_QWORD *)v44 )
  {
    v23 = v44;
    do
    {
      v21 = *(_QWORD *)(a2 + 32);
      (*(void (__fastcall **)(__int64 *, __int64, _QWORD))(a2 + 24))(&v39, v21, *v23);
      v29 = a1[1];
      if ( v29 == a1[2] )
      {
        v21 = a1[1];
        sub_EDB300(a1, (__int64 *)v21, &v39);
      }
      else
      {
        if ( v29 )
        {
          v24 = v39;
          *(_QWORD *)(v29 + 8) = 0;
          *(_QWORD *)(v29 + 16) = 0;
          *(_QWORD *)v29 = v24;
          v25 = v40;
          *(_BYTE *)(v29 + 24) = 0;
          v32 = v25;
          if ( v25 )
          {
            v26 = (__int64 *)sub_22077B0(32);
            if ( v26 )
            {
              v27 = v32;
              v33 = v26;
              *v26 = (__int64)(v26 + 2);
              v21 = *v27;
              sub_ED71E0(v26, (_BYTE *)*v27, *v27 + v27[1]);
              v26 = v33;
            }
            v28 = *(_QWORD **)(v29 + 8);
            *(_QWORD *)(v29 + 8) = v26;
            if ( v28 )
            {
              if ( (_QWORD *)*v28 != v28 + 2 )
              {
                v34 = v28;
                j_j___libc_free_0(*v28, v28[2] + 1LL);
                v28 = v34;
              }
              v21 = 32;
              j_j___libc_free_0(v28, 32);
            }
          }
          *(_DWORD *)(v29 + 16) = v41;
          *(_DWORD *)(v29 + 20) = v42;
          *(_BYTE *)(v29 + 24) = v43;
          v29 = a1[1];
        }
        a1[1] = v29 + 32;
      }
      v30 = v40;
      if ( v40 )
      {
        if ( (__int64 *)*v40 != v40 + 2 )
        {
          v35 = v40;
          j_j___libc_free_0(*v40, v40[2] + 1);
          v30 = v35;
        }
        v21 = 32;
        j_j___libc_free_0(v30, 32);
      }
      ++v23;
    }
    while ( v38 != v23 );
    v22 = v44;
  }
  if ( v22 != v46 )
    _libc_free(v22, v21);
  return a1;
}
