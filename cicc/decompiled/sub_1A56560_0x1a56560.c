// Function: sub_1A56560
// Address: 0x1a56560
//
__int64 __fastcall sub_1A56560(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  char *v6; // rax
  _BYTE *v7; // rsi
  int v8; // r8d
  __int64 *v9; // r15
  __int64 *v10; // r9
  __int64 *v11; // r15
  unsigned int v12; // edx
  __int64 v13; // rax
  __int64 v14; // r12
  char *v15; // rax
  char *v16; // rcx
  __int64 v17; // r14
  __int64 v18; // r12
  __int64 v19; // r15
  _BYTE *v20; // rsi
  int v21; // r9d
  __int64 v22; // r11
  __int64 v23; // r8
  __int64 v24; // r12
  char *v25; // rax
  __int64 v27; // [rsp+18h] [rbp-178h]
  __int64 v28; // [rsp+20h] [rbp-170h]
  __int64 *v29; // [rsp+20h] [rbp-170h]
  __int64 v30; // [rsp+28h] [rbp-168h]
  _QWORD *v31; // [rsp+38h] [rbp-158h] BYREF
  __int64 v32[2]; // [rsp+40h] [rbp-150h] BYREF
  char *v33; // [rsp+50h] [rbp-140h] BYREF
  __int64 v34; // [rsp+58h] [rbp-138h]
  _BYTE v35[304]; // [rsp+60h] [rbp-130h] BYREF

  v32[0] = a3;
  v32[1] = a4;
  v6 = (char *)sub_194ACF0(a4);
  v27 = (__int64)v6;
  if ( a2 )
  {
    v33 = v6;
    *(_QWORD *)v6 = a2;
    v7 = *(_BYTE **)(a2 + 16);
    if ( v7 == *(_BYTE **)(a2 + 24) )
    {
      sub_13FD960(a2 + 8, v7, &v33);
    }
    else
    {
      if ( v7 )
      {
        *(_QWORD *)v7 = v33;
        v7 = *(_BYTE **)(a2 + 16);
      }
      *(_QWORD *)(a2 + 16) = v7 + 8;
    }
  }
  else
  {
    v33 = v6;
    sub_1A541E0(a4 + 32, &v33);
  }
  sub_1A56230(v32, a1, v27);
  v9 = *(__int64 **)(a1 + 16);
  v10 = *(__int64 **)(a1 + 8);
  if ( v10 != v9 )
  {
    v11 = v9 - 1;
    v12 = 16;
    v33 = v35;
    v34 = 0x1000000000LL;
    v13 = 0;
    while ( 1 )
    {
      v14 = *v11;
      if ( (unsigned int)v13 >= v12 )
      {
        v29 = v10;
        sub_16CD150((__int64)&v33, v35, 0, 16, v8, (int)v10);
        v13 = (unsigned int)v34;
        v10 = v29;
      }
      v15 = &v33[16 * v13];
      *((_QWORD *)v15 + 1) = v14;
      *(_QWORD *)v15 = v27;
      v13 = (unsigned int)(v34 + 1);
      LODWORD(v34) = v34 + 1;
      if ( v10 == v11 )
        break;
      v12 = HIDWORD(v34);
      --v11;
    }
    do
    {
      while ( 1 )
      {
        v16 = &v33[16 * (unsigned int)v13 - 16];
        v17 = *(_QWORD *)v16;
        v18 = *((_QWORD *)v16 + 1);
        LODWORD(v34) = v13 - 1;
        v31 = sub_194ACF0(a4);
        v19 = (__int64)v31;
        *v31 = v17;
        v20 = *(_BYTE **)(v17 + 16);
        if ( v20 == *(_BYTE **)(v17 + 24) )
        {
          sub_13FD960(v17 + 8, v20, &v31);
        }
        else
        {
          if ( v20 )
          {
            *(_QWORD *)v20 = v31;
            v20 = *(_BYTE **)(v17 + 16);
          }
          *(_QWORD *)(v17 + 16) = v20 + 8;
        }
        sub_1A56230(v32, v18, v19);
        v22 = *(_QWORD *)(v18 + 8);
        v23 = *(_QWORD *)(v18 + 16);
        if ( v22 != v23 )
          break;
        LODWORD(v13) = v34;
        if ( !(_DWORD)v34 )
          goto LABEL_22;
      }
      v13 = (unsigned int)v34;
      do
      {
        v24 = *(_QWORD *)(v23 - 8);
        if ( HIDWORD(v34) <= (unsigned int)v13 )
        {
          v28 = v23;
          v30 = v22;
          sub_16CD150((__int64)&v33, v35, 0, 16, v23, v21);
          v13 = (unsigned int)v34;
          v23 = v28;
          v22 = v30;
        }
        v25 = &v33[16 * v13];
        v23 -= 8;
        *(_QWORD *)v25 = v19;
        *((_QWORD *)v25 + 1) = v24;
        v13 = (unsigned int)(v34 + 1);
        LODWORD(v34) = v34 + 1;
      }
      while ( v22 != v23 );
    }
    while ( (_DWORD)v13 );
LABEL_22:
    if ( v33 != v35 )
      _libc_free((unsigned __int64)v33);
  }
  return v27;
}
