// Function: sub_DCF0D0
// Address: 0xdcf0d0
//
__int64 __fastcall sub_DCF0D0(__int64 *a1, __int64 a2, __int64 *a3, __int64 a4)
{
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // rbx
  __int64 v10; // rax
  __int64 v11; // r13
  __int64 v12; // r9
  _BYTE *v13; // rdx
  __int64 v14; // rbx
  __int64 v15; // rax
  unsigned __int64 v16; // rdx
  _QWORD *v17; // r12
  __int64 v18; // [rsp+0h] [rbp-60h]
  _BYTE *v19; // [rsp+10h] [rbp-50h] BYREF
  __int64 v20; // [rsp+18h] [rbp-48h]
  _BYTE v21[64]; // [rsp+20h] [rbp-40h] BYREF

  if ( !*((_BYTE *)a1 + 136) || !*((_DWORD *)a1 + 2) || !sub_D47930(a2) )
    return sub_D970F0((__int64)a3);
  v9 = *a1;
  v20 = 0x200000000LL;
  v10 = *((unsigned int *)a1 + 2);
  v19 = v21;
  v11 = v9 + 112 * v10;
  if ( v9 != v11 )
  {
    v12 = *(_QWORD *)(v9 + 40);
    v13 = v21;
    v14 = v9 + 112;
    v15 = 0;
    while ( 1 )
    {
      *(_QWORD *)&v13[8 * v15] = v12;
      LODWORD(v20) = v20 + 1;
      if ( a4 )
        sub_D91A50(
          a4,
          (char *)(*(_QWORD *)a4 + 8LL * *(unsigned int *)(a4 + 8)),
          *(char **)(v14 - 48),
          (char *)(*(_QWORD *)(v14 - 48) + 8LL * *(unsigned int *)(v14 - 40)));
      if ( v11 == v14 )
        break;
      v15 = (unsigned int)v20;
      v7 = HIDWORD(v20);
      v12 = *(_QWORD *)(v14 + 40);
      v16 = (unsigned int)v20 + 1LL;
      if ( v16 > HIDWORD(v20) )
      {
        v18 = *(_QWORD *)(v14 + 40);
        sub_C8D5F0((__int64)&v19, v21, v16, 8u, v8, v12);
        v15 = (unsigned int)v20;
        v12 = v18;
      }
      v13 = v19;
      v14 += 112;
    }
  }
  v17 = sub_DCEEE0(a3, (__int64)&v19, 1, v7, v8);
  if ( v19 != v21 )
    _libc_free(v19, &v19);
  return (__int64)v17;
}
