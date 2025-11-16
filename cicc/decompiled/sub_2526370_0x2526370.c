// Function: sub_2526370
// Address: 0x2526370
//
__int64 __fastcall sub_2526370(
        __int64 a1,
        __int64 (__fastcall *a2)(__int64, unsigned __int64, __int64),
        __int64 a3,
        __int64 a4,
        int *a5,
        __int64 a6,
        _BYTE *a7,
        char a8,
        unsigned __int8 a9)
{
  __int64 *v10; // rdi
  __int64 v14; // rcx
  __int64 v15; // rax
  unsigned __int8 *v16; // rcx
  int v17; // eax
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // r10
  unsigned __int64 v21; // rcx
  unsigned __int64 v23; // rax
  __int64 v24; // r8
  unsigned __int64 v25; // rax
  unsigned __int8 *v26; // rax
  __int64 v29; // [rsp+10h] [rbp-40h]
  __int64 v30; // [rsp+18h] [rbp-38h]
  unsigned __int8 **v31; // [rsp+18h] [rbp-38h]

  v10 = (__int64 *)(a4 + 72);
  v14 = *(_QWORD *)(a4 + 72);
  v15 = v14 & 3;
  v16 = (unsigned __int8 *)(v14 & 0xFFFFFFFFFFFFFFFCLL);
  if ( v15 == 3 )
    v16 = (unsigned __int8 *)*((_QWORD *)v16 + 3);
  v17 = *v16;
  if ( (unsigned __int8)v17 > 0x1Cu
    && (v23 = (unsigned int)(v17 - 34), (unsigned __int8)v23 <= 0x33u)
    && (v24 = 0x8000000000041LL, _bittest64(&v24, v23)) )
  {
    v29 = a1;
    v31 = (unsigned __int8 **)v16;
    v25 = sub_250C680(v10);
    v20 = v29;
    v19 = a3;
    if ( v25 )
    {
      v21 = *(_QWORD *)(v25 + 24);
    }
    else
    {
      v26 = sub_BD3990(*(v31 - 4), (__int64)a2);
      v20 = v29;
      v19 = a3;
      v21 = (unsigned __int64)v26;
      if ( v26 && *v26 )
        v21 = 0;
    }
  }
  else
  {
    v30 = a1;
    v18 = sub_25096F0(v10);
    v19 = a3;
    v20 = v30;
    v21 = v18;
  }
  return sub_2526260(v20, a2, v19, v21, a4, a7, a5, a6, a8, a9);
}
