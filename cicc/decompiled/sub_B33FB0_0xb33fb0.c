// Function: sub_B33FB0
// Address: 0xb33fb0
//
__int64 __fastcall sub_B33FB0(__int64 a1, __int64 a2, __int64 a3)
{
  int v4; // edx
  __int64 *v5; // rax
  __int64 v6; // r14
  __int64 v7; // r12
  __int64 v8; // rbx
  __int64 v9; // r15
  __int64 v10; // rax
  unsigned __int64 v11; // rdx
  __int64 v12; // rsi
  __int64 *v13; // rdi
  __int64 v14; // r14
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // r14
  __int64 v19; // rax
  __int64 v20; // rbx
  unsigned int *v21; // rbx
  __int64 v22; // r12
  __int64 v23; // rdx
  __int64 v24; // rsi
  __int64 v25; // [rsp-8h] [rbp-D8h]
  __int64 v26; // [rsp+18h] [rbp-B8h]
  _QWORD v27[4]; // [rsp+20h] [rbp-B0h] BYREF
  __int16 v28; // [rsp+40h] [rbp-90h]
  __int64 *v29; // [rsp+50h] [rbp-80h] BYREF
  __int64 v30; // [rsp+58h] [rbp-78h]
  _BYTE v31[16]; // [rsp+60h] [rbp-70h] BYREF
  __int16 v32; // [rsp+70h] [rbp-60h]

  v4 = *(unsigned __int8 *)(a2 + 8);
  if ( (unsigned int)(v4 - 17) > 1 )
  {
    v6 = a2;
    goto LABEL_3;
  }
  v5 = *(__int64 **)(a2 + 16);
  v6 = *v5;
  if ( (_BYTE)v4 != 18 )
  {
LABEL_3:
    v7 = *(unsigned int *)(a2 + 32);
    v29 = (__int64 *)v31;
    v30 = 0x800000000LL;
    if ( (_DWORD)v7 )
    {
      v8 = 0;
      do
      {
        v9 = sub_AD64C0(v6, v8, 0);
        v10 = (unsigned int)v30;
        v11 = (unsigned int)v30 + 1LL;
        if ( v11 > HIDWORD(v30) )
        {
          sub_C8D5F0(&v29, v31, v11, 8);
          v10 = (unsigned int)v30;
        }
        ++v8;
        v29[v10] = v9;
        v12 = (unsigned int)(v30 + 1);
        LODWORD(v30) = v30 + 1;
      }
      while ( v7 != v8 );
      v13 = v29;
    }
    else
    {
      v12 = 0;
      v13 = (__int64 *)v31;
    }
    v14 = sub_AD3730(v13, v12);
    if ( v29 != (__int64 *)v31 )
      _libc_free(v29, v12);
    return v14;
  }
  if ( (unsigned int)sub_BCB060(*v5) <= 7 )
  {
    v16 = sub_BCB2B0(*(_QWORD *)(a1 + 72));
    BYTE4(v26) = *(_BYTE *)(a2 + 8) == 18;
    LODWORD(v26) = *(_DWORD *)(a2 + 32);
    v17 = sub_BCE1B0(v16, v26);
    BYTE4(v29) = 0;
    v18 = v17;
    v27[0] = v17;
    v19 = sub_B33D10(a1, 0x159u, (__int64)v27, 1, 0, 0, (__int64)v29, a3);
    v20 = v19;
    if ( a2 == v18 )
      return v19;
    v28 = 257;
    if ( a2 == *(_QWORD *)(v19 + 8) )
    {
      return v19;
    }
    else
    {
      v14 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64, __int64))(**(_QWORD **)(a1 + 80) + 120LL))(
              *(_QWORD *)(a1 + 80),
              38,
              v19,
              a2,
              v25);
      if ( !v14 )
      {
        v32 = 257;
        v14 = sub_B51D30(38, v20, a2, &v29, 0, 0);
        (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a1 + 88) + 16LL))(
          *(_QWORD *)(a1 + 88),
          v14,
          v27,
          *(_QWORD *)(a1 + 56),
          *(_QWORD *)(a1 + 64));
        v21 = *(unsigned int **)a1;
        v22 = *(_QWORD *)a1 + 16LL * *(unsigned int *)(a1 + 8);
        if ( *(_QWORD *)a1 != v22 )
        {
          do
          {
            v23 = *((_QWORD *)v21 + 1);
            v24 = *v21;
            v21 += 4;
            sub_B99FD0(v14, v24, v23);
          }
          while ( (unsigned int *)v22 != v21 );
        }
      }
    }
  }
  else
  {
    BYTE4(v29) = 0;
    v27[0] = a2;
    return sub_B33D10(a1, 0x159u, (__int64)v27, 1, 0, 0, (__int64)v29, a3);
  }
  return v14;
}
