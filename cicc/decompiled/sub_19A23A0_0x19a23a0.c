// Function: sub_19A23A0
// Address: 0x19a23a0
//
void __fastcall sub_19A23A0(
        __int64 a1,
        __int64 a2,
        unsigned int a3,
        __int64 *a4,
        int a5,
        int a6,
        __m128i a7,
        __m128i a8)
{
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // rax
  __int64 v12; // rax
  unsigned __int64 v13; // rdi
  __int64 v14; // rdx
  __int64 *v15; // r8
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 *v18; // rbx
  __int64 *v19; // r14
  __int64 v20; // rax
  __int64 v21; // r15
  int v22; // r8d
  int v23; // r9d
  __int64 v24; // rax
  int v25; // r8d
  int v26; // r9d
  __int64 *v28; // [rsp+38h] [rbp-C8h] BYREF
  _BYTE *v29; // [rsp+40h] [rbp-C0h] BYREF
  __int64 v30; // [rsp+48h] [rbp-B8h]
  _BYTE v31[32]; // [rsp+50h] [rbp-B0h] BYREF
  _QWORD v32[2]; // [rsp+70h] [rbp-90h] BYREF
  char v33; // [rsp+80h] [rbp-80h]
  __int64 v34; // [rsp+88h] [rbp-78h]
  _BYTE *v35; // [rsp+90h] [rbp-70h] BYREF
  __int64 v36; // [rsp+98h] [rbp-68h]
  _BYTE v37[32]; // [rsp+A0h] [rbp-60h] BYREF
  __int64 v38; // [rsp+C0h] [rbp-40h]
  __int64 v39; // [rsp+C8h] [rbp-38h]

  v9 = a4[3];
  if ( *((unsigned int *)a4 + 10) + (unsigned __int64)(v9 == 1) > 1 )
  {
    v10 = *((unsigned int *)a4 + 10);
    if ( v9 == 1 )
    {
      a4[3] = 0;
      sub_1458920((__int64)(a4 + 4), a4 + 10);
      a4[10] = 0;
      v9 = a4[3];
      v10 = *((unsigned int *)a4 + 10);
    }
    v11 = *a4;
    v34 = v9;
    v35 = v37;
    v32[0] = v11;
    v12 = a4[1];
    v36 = 0x400000000LL;
    v32[1] = v12;
    v33 = *((_BYTE *)a4 + 16);
    if ( !(_DWORD)v10 )
      goto LABEL_5;
    sub_19930D0((__int64)&v35, (__int64)(a4 + 4), v9, v10, a5, a6);
    v14 = a4[10];
    v15 = (__int64 *)a4[4];
    v29 = v31;
    v16 = *((unsigned int *)a4 + 10);
    v38 = v14;
    v17 = a4[11];
    v18 = &v15[v16];
    LODWORD(v36) = 0;
    v39 = v17;
    v30 = 0x400000000LL;
    if ( v18 == v15 )
      goto LABEL_5;
    v19 = v15;
    do
    {
      while ( 1 )
      {
        v21 = *v19;
        if ( !sub_146D930(*(_QWORD *)(a1 + 8), *v19, **(_QWORD **)(*(_QWORD *)(a1 + 40) + 32LL))
          || sub_146D100(*(_QWORD *)(a1 + 8), v21, *(_QWORD *)(a1 + 40)) )
        {
          break;
        }
        v24 = (unsigned int)v30;
        if ( (unsigned int)v30 >= HIDWORD(v30) )
        {
          sub_16CD150((__int64)&v29, v31, 0, 8, v22, v23);
          v24 = (unsigned int)v30;
        }
        ++v19;
        *(_QWORD *)&v29[8 * v24] = v21;
        LODWORD(v30) = v30 + 1;
        if ( v18 == v19 )
          goto LABEL_18;
      }
      v20 = (unsigned int)v36;
      if ( (unsigned int)v36 >= HIDWORD(v36) )
      {
        sub_16CD150((__int64)&v35, v37, 0, 8, v22, v23);
        v20 = (unsigned int)v36;
      }
      ++v19;
      *(_QWORD *)&v35[8 * v20] = v21;
      LODWORD(v36) = v36 + 1;
    }
    while ( v18 != v19 );
LABEL_18:
    if ( (unsigned int)v30 > 1 )
    {
      v28 = sub_147DD40(*(_QWORD *)(a1 + 8), (__int64 *)&v29, 0, 0, a7, a8);
      if ( !sub_14560B0((__int64)v28) )
      {
        sub_1458920((__int64)&v35, &v28);
        sub_19932F0((__int64)v32, *(_QWORD *)(a1 + 40));
        sub_19A1660(a1, a2, a3, (__int64)v32, v25, v26);
      }
    }
    if ( v29 == v31 )
    {
LABEL_5:
      v13 = (unsigned __int64)v35;
      if ( v35 == v37 )
        return;
    }
    else
    {
      _libc_free((unsigned __int64)v29);
      v13 = (unsigned __int64)v35;
      if ( v35 == v37 )
        return;
    }
    _libc_free(v13);
  }
}
