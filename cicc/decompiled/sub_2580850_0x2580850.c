// Function: sub_2580850
// Address: 0x2580850
//
__int64 __fastcall sub_2580850(__int64 a1, __int64 a2, __m128i *a3, _DWORD *a4, char *a5, char a6)
{
  unsigned int v8; // r15d
  unsigned __int8 **v9; // rdi
  __int64 v11; // rcx
  unsigned __int8 **v12; // rcx
  unsigned __int8 **v13; // rbx
  int v14; // edx
  char v15; // dl
  __int64 v16; // rsi
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rbx
  __int64 (__fastcall *v20)(__int64); // rax
  unsigned __int8 *v21; // rdi
  __int64 (__fastcall *v22)(__int64); // rax
  __int64 (__fastcall *v23)(__int64); // rax
  __int64 v24; // rax
  __int64 (__fastcall *v25)(__int64); // rax
  __int64 v26; // rbx
  __int64 v27; // [rsp-8h] [rbp-B0h]
  __int64 v28; // [rsp-8h] [rbp-B0h]
  unsigned __int8 **v31; // [rsp+18h] [rbp-90h]
  char v33; // [rsp+37h] [rbp-71h] BYREF
  unsigned __int8 **v34; // [rsp+38h] [rbp-70h] BYREF
  __int64 v35; // [rsp+40h] [rbp-68h]
  _BYTE v36[96]; // [rsp+48h] [rbp-60h] BYREF

  v34 = (unsigned __int8 **)v36;
  v35 = 0x300000000LL;
  v33 = 0;
  v8 = sub_2526B50(a2, a3, a1, (__int64)&v34, 2u, &v33, 1u);
  if ( (_BYTE)v8 )
  {
    v11 = (unsigned int)v35;
    v9 = v34;
    *a5 = 0;
    v12 = &v9[2 * v11];
    if ( v12 != v9 )
    {
      v13 = v9;
      while ( 1 )
      {
        v14 = **v13;
        if ( (unsigned int)(v14 - 12) > 1 )
        {
          if ( (_BYTE)v14 != 17 )
          {
            v9 = v34;
            v8 = 0;
            goto LABEL_4;
          }
          v31 = v12;
          sub_2575FB0(a4, (const void **)*v13 + 3);
          v12 = v31;
        }
        else
        {
          *a5 = 1;
        }
        v13 += 2;
        if ( v12 == v13 )
        {
          v15 = *a5;
          v9 = v34;
          goto LABEL_13;
        }
      }
    }
    v15 = 0;
LABEL_13:
    *a5 = v15 & (a4[10] == 0);
  }
  else
  {
    if ( !a6 && *(_BYTE *)(sub_250D180(a3->m128i_i64, v27) + 8) == 12 )
    {
      v16 = a3->m128i_i64[0];
      v17 = sub_25803A0(a2, a3->m128i_i64[0], a3->m128i_i64[1], a1, 0, 0, 1);
      v18 = v28;
      v19 = v17;
      if ( v17 )
      {
        v20 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v17 + 48LL);
        v21 = (unsigned __int8 *)(v20 == sub_2534B10
                                ? v19 + 88
                                : ((__int64 (__fastcall *)(__int64, __int64, __int64))v20)(v19, v16, v28));
        v22 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v21 + 16LL);
        v8 = v22 == sub_2505E40
           ? v21[17]
           : ((__int64 (__fastcall *)(unsigned __int8 *, __int64, __int64))v22)(v21, v16, v18);
        if ( (_BYTE)v8 )
        {
          v23 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v19 + 48LL);
          if ( v23 == sub_2534B10 )
            v24 = v19 + 88;
          else
            v24 = ((__int64 (__fastcall *)(__int64, __int64, __int64))v23)(v19, v16, v18);
          *a5 = *(_BYTE *)(v24 + 200);
          v25 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v19 + 48LL);
          if ( v25 == sub_2534B10 )
            v26 = v19 + 88;
          else
            v26 = ((__int64 (__fastcall *)(__int64, __int64, __int64))v25)(v19, v16, v18);
          if ( (_DWORD *)(v26 + 24) != a4 )
            sub_255EE30((__int64)a4, v26 + 24);
          sub_2560D30(a4 + 8, v26 + 56);
        }
      }
    }
    v9 = v34;
  }
LABEL_4:
  if ( v9 != (unsigned __int8 **)v36 )
    _libc_free((unsigned __int64)v9);
  return v8;
}
