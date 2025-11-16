// Function: sub_1BDC800
// Address: 0x1bdc800
//
__int64 __fastcall sub_1BDC800(
        __int64 a1,
        __int64 *a2,
        __int64 a3,
        __int64 a4,
        __m128i a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        double a11,
        __m128 a12)
{
  unsigned int v12; // r15d
  __int64 *v14; // rbx
  __int64 v15; // rax
  __int64 v16; // rax
  int v17; // r8d
  int v18; // r9d
  double v19; // xmm4_8
  double v20; // xmm5_8
  __int64 **v22; // r15
  __int64 ***v23; // rdx
  __int64 v24; // rax
  char v25; // dl
  __int64 v26; // rdx
  __int64 ***v27; // rcx
  unsigned __int64 v28; // rdx
  __int64 ***v29; // rsi
  __int64 ***v30; // rax
  __int64 **v31; // rdx
  __int64 **v32; // rsi
  __int64 ***v33; // [rsp+10h] [rbp-C0h] BYREF
  __int64 v34; // [rsp+18h] [rbp-B8h]
  _BYTE v35[176]; // [rsp+20h] [rbp-B0h] BYREF

  v12 = 0;
  v14 = a2;
  v15 = sub_157EB90(a3);
  v16 = sub_1632FA0(v15);
  if ( (unsigned int)sub_1BBCE00(a4, *a2, v16) )
  {
    v22 = (__int64 **)*(a2 - 3);
    v33 = (__int64 ***)v35;
    v23 = (__int64 ***)v35;
    v34 = 0x1000000000LL;
    v24 = 0;
    while ( 1 )
    {
      v23[v24] = v22;
      v24 = (unsigned int)(v34 + 1);
      LODWORD(v34) = v34 + 1;
      v14 = (__int64 *)*(v14 - 6);
      v25 = *((_BYTE *)v14 + 16);
      if ( v25 == 9 )
        break;
      if ( v25 != 87 || (v26 = v14[1]) == 0 || *(_QWORD *)(v26 + 8) )
      {
        v12 = 0;
        goto LABEL_7;
      }
      v22 = (__int64 **)*(v14 - 3);
      if ( HIDWORD(v34) <= (unsigned int)v24 )
      {
        sub_16CD150((__int64)&v33, v35, 0, 8, v17, v18);
        v24 = (unsigned int)v34;
      }
      v23 = v33;
    }
    v27 = v33;
    v28 = (unsigned int)v24;
    v29 = &v33[(unsigned int)v24];
    if ( v33 != v29 )
    {
      v30 = v29 - 1;
      if ( v33 >= v29 - 1 )
      {
        v29 = v33;
      }
      else
      {
        do
        {
          v31 = *v27;
          v32 = *v30;
          ++v27;
          --v30;
          *(v27 - 1) = v32;
          v30[1] = v31;
        }
        while ( v27 < v30 );
        v29 = v33;
        v28 = (unsigned int)v34;
      }
    }
    v12 = sub_1BDB410(a1, v29, v28, a4, 0, 0, a5, a6, a7, a8, v19, v20, a11, a12);
LABEL_7:
    if ( v33 != (__int64 ***)v35 )
      _libc_free((unsigned __int64)v33);
  }
  return v12;
}
